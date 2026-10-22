"""The edit list, and the song it leaves behind: parsing, refusals, the grid after an edit, the names of results.

A track window writes a JSON list and the node reads it, so the shape of that
list is checked here rather than in the browser. The rest is the chain: a
second edit is made on the song the first one left, which is shorter than the
one that was sung, and its bars have moved. Nothing here needs torch.
"""

from __future__ import annotations

import json
import types

import pytest

from yue2_comfy import notation
from yue2_comfy.inpaint import grid, ops, track

HEADER = ('X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n'
          'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
          'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n')


def score(*sections):
    """A score of ``(name, bars)`` sections, the Ins part resting, at most four bars a line."""
    lines = [HEADER.rstrip("\n")]
    for name, bars in sections:
        lines.append("% " + name)
        for start in range(0, len(bars), 4):
            group = bars[start:start + 4]
            lines += ["V: Vocal", "|".join(group) + "|", "V: Ins", "|".join(["Z"] * len(group)) + "|"]
    return "\n".join(lines) + "\n"


RAP = score(("intro", ["z16", "z8z2DDA2AA"]),
            ("verse", ["A2DDAAA2AGGGG2GG", "AGGGGFFFFDD2B4"]),
            ("chorus", ["A2z8D2B4", "A2z4FG3F2B2B2", "A2z8DDBB3", "A2z8z2G2G2"]),
            ("outro", ["z16", "z16"]))
"""The grid tests' rap: intro 0-1, verse 2-3, chorus 4-7, outro 8-9, every bar two seconds long."""

LYRICS = "[Intro]\n\n[Verse]\none two three\nfour five six\n\n[Chorus]\nseven eight\n\n[Outro]"

FRAMES = 500
"""Twenty seconds: the ten bars from 0.5 s, and a little after them."""


def rap_song(score_text=RAP, lyrics=LYRICS):
    return types.SimpleNamespace(score=score_text, lyrics=lyrics, frames=FRAMES)


def rap_state(offset=0.5):
    sheet = notation.read(RAP)
    clock = grid.Grid(offset=offset, rate=1.0, tick=grid.tick_seconds(sheet),
                      starts=grid.starts_of(sheet))
    return track.opened(rap_song(), clock)


def test_an_empty_field_is_no_edits_at_all():
    """A node whose track has never been opened runs, hands the song on and draws the grid."""
    assert track.read("") == ()
    assert track.read("   \n") == ()
    assert track.read(None) == ()


def test_an_edit_list_reads_as_the_window_wrote_it():
    text = ('[{"op": "retake", "bars": [4, 8], "seed": 831001, "takes": 3, "take": 2},'
            ' {"op": "cut", "bars": [2, 4]},'
            ' {"op": "retake", "seconds": [1.5, 4.0], "seed": 7}]')
    first, second, third = track.read(text, takes=2)
    assert (first.op, first.bars, first.seconds) == ("retake", (4, 8), None)
    assert (first.seed, first.takes, first.take) == (831001, 3, 2)
    assert first.seeds() == (831001, 831002, 831003)
    assert (second.op, second.bars, second.takes, second.take) == ("cut", (2, 4), 1, None)
    assert (third.op, third.seconds, third.takes) == ("retake", (1.5, 4.0), 2)
    assert track.read(track.written((first, second, third)), takes=2) == (first, second, third)


def test_a_list_may_name_the_song_it_was_made_on():
    """The window writes {"song": key, "edits": [...]}; a bare list, as every older one is, names none."""
    edits = '[{"op": "cut", "bars": [2, 4]}]'
    named = '{"song": "%s", "edits": %s}' % ("ab" * 32, edits)
    assert track.read(named) == track.read(edits)
    assert track.song_of(named) == "ab" * 32
    for text in (edits, "", None, "not json", '{"edits": []}', '{"song": 5, "edits": []}'):
        assert track.song_of(text) == "", text
    for text in ('{"song": 5, "edits": []}', '{"song": "x"}', '{"song": "x", "edits": {}}'):
        with pytest.raises(ValueError, match="does not hold a JSON list"):
            track.read(text)


def test_an_edit_list_that_is_not_one_is_refused_in_words_a_person_can_act_on():
    for text in ("{}", "[1, 2]", "not json", '"a string"'):
        with pytest.raises(ValueError):
            track.read(text)
    problems = [
        ('[{"op": "sing", "bars": [0, 1]}]', "not something an edit does"),
        ('[{"op": "retake"}]', "neither bars nor seconds"),
        ('[{"op": "retake", "bars": [0, 1], "seconds": [0, 1]}]', "neither bars nor seconds"),
        ('[{"op": "retake", "bars": [0]}]', "not a pair"),
        ('[{"op": "retake", "bars": [4, 4]}]', "not a stretch of a score"),
        ('[{"op": "retake", "bars": [4, 2]}]', "not a stretch of a score"),
        ('[{"op": "retake", "bars": [0.5, 2]}]', "not a whole number"),
        ('[{"op": "retake", "seconds": [4.0, 1.0]}]', "not a stretch of a song"),
        ('[{"op": "retake", "seconds": ["a", 1.0]}]', "not a number"),
        ('[{"op": "cut", "bars": [0, 2]}, {"op": "retake", "bars": [0, 2], "takes": 9}]',
         "between 1 and 4"),
        ('[{"op": "retake", "bars": [0, 2], "takes": 2, "take": 5}]', "keeps take 6 of 2"),
        ('[{"op": "retake", "bars": [0, 2], "seed": "x"}]', "not a whole number"),
    ]
    for text, said in problems:
        with pytest.raises(ValueError, match=said):
            track.read(text)


def test_an_edit_says_which_one_it_is_in_its_refusal():
    """A list is written by a window and read by a node, so a person has to be told which edit is wrong."""
    with pytest.raises(ValueError, match="Edit 2 "):
        track.read('[{"op": "cut", "bars": [0, 2]}, {"op": "cut"}]')


def test_a_retake_of_bars_takes_the_frames_the_grid_gives_it_and_leaves_the_words_alone():
    state = rap_state()
    step = track.plan(state, track.read('[{"op": "retake", "bars": [4, 8]}]')[0])
    assert (step.start, step.stop) == (192, 414)
    assert (step.kind, step.bars, step.dropped) == ("retake", (4, 8), ())
    assert (step.lyrics, step.score) == (LYRICS, RAP)


def test_a_cut_of_bars_takes_them_out_of_the_score_and_the_words_with_them():
    state = rap_state()
    step = track.plan(state, track.read('[{"op": "cut", "bars": [2, 4]}]')[0])
    assert (step.start, step.stop) == (92, 192)
    assert step.dropped == ("Verse",)
    assert "one two three" not in step.lyrics
    assert "seven eight" in step.lyrics
    assert len(notation.read(step.score)["bars"]) == 8
    assert step.notices and step.notices[0][0] == "notice"


def test_a_song_without_a_score_is_edited_by_the_second_and_cut_by_it_too():
    """'cot' off leaves no score and so no bars; the window then selects seconds."""
    state = track.opened(types.SimpleNamespace(score="", lyrics="la la", frames=FRAMES), None)
    step = track.plan(state, track.read('[{"op": "retake", "seconds": [2.0, 6.0]}]')[0])
    assert (step.start, step.stop) == (50, 150)
    cut = track.plan(state, track.read('[{"op": "cut", "seconds": [2.0, 6.0]}]')[0])
    assert (cut.start, cut.stop, cut.score, cut.dropped) == (50, 150, "", ())
    assert cut.lyrics == "la la"
    with pytest.raises(ValueError, match="no score and no bars"):
        track.plan(state, track.read('[{"op": "retake", "bars": [0, 2]}]')[0])


def test_a_cut_of_a_song_that_has_a_score_goes_by_bars():
    """Seconds would leave the score and the words describing a song that is no longer there."""
    with pytest.raises(ValueError, match="goes by whole bars"):
        track.plan(rap_state(), track.read('[{"op": "cut", "seconds": [2.0, 6.0]}]')[0])


def test_seconds_outside_the_song_are_refused_with_its_length():
    state = rap_state()
    with pytest.raises(ValueError, match="20.00 seconds long"):
        track.plan(state, track.read('[{"op": "retake", "seconds": [30.0, 40.0]}]')[0])


def test_the_bars_of_the_song_a_cut_leaves_are_where_the_sound_now_has_them():
    """Cutting the rap's verse takes bars 2 and 3 and the hundred frames they were: everything after
    comes back four seconds, and the chorus begins where the verse did."""
    state = rap_state()
    step = track.plan(state, track.read('[{"op": "cut", "bars": [2, 4]}]')[0])
    after = track.after(state, step, 0)
    assert after.frames == FRAMES - 100
    assert len(after.sheet["bars"]) == 8
    assert after.clock.at(0) == pytest.approx(0.5)
    assert after.clock.at(2) == pytest.approx(4.5)
    assert after.clock.at(6) == pytest.approx(12.5)


def test_the_bars_after_a_retake_move_by_what_it_sang_and_the_ones_before_it_stay():
    """A retake of the chorus that comes out a second short pulls the end of the song a second
    earlier. The chorus opens at its pickup, frame 192, and closes past the outro's line, frame
    414, so its own bar lines are inside the new singing and are spread through it; the score
    keeps them, nobody knows where the new take put them."""
    state = rap_state()
    step = track.plan(state, track.read('[{"op": "retake", "bars": [4, 8]}]')[0])
    assert (step.start, step.stop) == (192, 414)
    after = track.after(state, step, (step.stop - step.start) - 25)
    assert after.frames == FRAMES - 25
    assert after.sheet is state.sheet
    assert after.clock.at(2) == pytest.approx(4.5)
    assert after.clock.at(4) == pytest.approx(8.5 - 0.09, abs=0.01)
    assert after.clock.at(8) == pytest.approx(15.5, abs=0.01)
    assert after.clock.at(10) == pytest.approx(19.5)


def test_a_retake_that_came_out_as_long_as_it_was_leaves_the_grid_alone():
    state = rap_state()
    step = track.plan(state, track.read('[{"op": "retake", "bars": [4, 8]}]')[0])
    assert track.after(state, step, step.stop - step.start).clock is state.clock


def test_a_second_edit_is_made_on_the_song_the_first_one_left():
    """After the verse is cut, the chorus is bars 2 to 6 and stands a hundred frames earlier, so
    its retake ends at 414 - 100. It opens earlier than it would have: the rap ran straight into
    the chorus and the seam was the margin before its pickup, and now the intro's rest lies
    before it, so the seam falls a beat before the pickup instead -- ten sixteenths before bar 2
    at 4.5 s, frame 81. The eleven frames between are the intro's rest, and singing them again
    costs nothing."""
    state = rap_state()
    first = track.plan(state, track.read('[{"op": "cut", "bars": [2, 4]}]')[0])
    assert (first.start, first.stop) == (92, 192)
    state = track.after(state, first, 0)
    second = track.plan(state, track.read('[{"op": "retake", "bars": [2, 6]}]')[0])
    assert (second.start, second.stop) == (81, 314)


def test_a_cut_that_runs_into_the_end_of_the_song_ends_the_bars_there():
    """A song can be shorter than its score. A cut of the last bars then takes less than their own
    length, and the lines after it cannot come back past where the cut began: the grid used to put
    the last line before the one before it (6.98 s after 14.5 s on this song of 30 s)."""
    sheet = notation.read(RAP)
    clock = grid.Grid(offset=0.5, rate=1.0, tick=grid.tick_seconds(sheet), starts=grid.starts_of(sheet))
    state = track.opened(types.SimpleNamespace(score=RAP, lyrics=LYRICS, frames=750), clock)
    step = track.plan(state, track.read('[{"op": "cut", "bars": [8, 10]}]')[0])
    assert (step.start, step.stop) == (412, 750)
    after = track.after(state, step, 0)
    assert after.frames == 412
    assert list(after.clock.lines) == pytest.approx([0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.48])
    assert len(after.clock.lines) == len(after.clock.starts) == len(after.sheet["bars"]) + 1


def test_a_cut_that_reaches_the_end_of_the_song_takes_the_bars_the_song_never_sang():
    """The song of 20 s stops inside bar 9 of 10. A cut of bars 4 to 9 takes the rest of the song,
    so the score loses bar 10 too, and says so. The score's last line stood half a second past
    the end of the song before the cut, and it still does after it."""
    state = rap_state()
    step = track.plan(state, track.read('[{"op": "cut", "bars": [3, 9]}]')[0])
    assert (step.start, step.stop) == (212, 500)
    assert step.bars == (3, 10)
    assert any("bars 10 to 10 go with it" in text for _kind, text in step.notices)
    after = track.after(state, step, 0)
    assert len(after.sheet["bars"]) == 3
    assert list(after.clock.lines) == pytest.approx([0.5, 2.5, 4.5, 8.98])
    assert after.frames == 212


def test_the_beats_of_an_edited_bar_follow_its_own_length():
    """A retake that came out short makes the bars inside it shorter, and their beats with them."""
    state = rap_state()
    step = track.plan(state, track.read('[{"op": "retake", "bars": [4, 8]}]')[0])
    after = track.after(state, step, (step.stop - step.start) - 25)
    drawn = grid.layout(after.sheet, after.clock, after.frames)
    bar4, bar5 = drawn["bars"][4], drawn["bars"][5]
    beats = [beat for beat in drawn["beats"] if bar4 <= beat < bar5]
    assert len(beats) == 4
    assert beats[1] - beats[0] == pytest.approx((bar5 - bar4) / 4, abs=0.002)
    assert beats[3] < bar5


def test_infinity_is_not_a_second():
    """json reads Infinity; a selection of it would fall over as an OverflowError past every refusal."""
    for text in ('[{"op": "retake", "seconds": [1.0, Infinity]}]',
                 '[{"op": "retake", "seconds": [-Infinity, 1.0]}]',
                 '[{"op": "retake", "seconds": [NaN, 1.0]}]'):
        with pytest.raises(ValueError, match="not a number"):
            track.read(text)


def test_a_cut_reads_the_same_whatever_seed_or_take_it_was_written_with():
    """A cut has no seed and no take; one written with them must not get another name after a
    round trip through the field, or every take after it would be sung again."""
    plain = track.read('[{"op": "cut", "bars": [2, 4]}]')[0]
    dressed = track.read('[{"op": "cut", "bars": [2, 4], "seed": 42, "take": 0, "takes": 3}]')[0]
    assert plain == dressed
    assert track.name("song", [], plain) == track.name("song", [], dressed)
    assert track.read(track.written((dressed,))) == (plain,)


def test_the_words_before_the_first_tag_are_named_when_a_cut_takes_them():
    lyrics = "la la la\n\n[Verse]\none two three\nfour five six\n\n[Chorus]\nseven eight\n\n[Outro]"
    state = track.opened(rap_song(lyrics=lyrics), rap_state().clock)
    step = track.plan(state, track.read('[{"op": "cut", "bars": [0, 2]}]')[0])
    assert step.dropped == ("",)
    assert "the lines before the first tag" in step.notices[0][1]


def test_a_retake_may_ask_for_its_own_sampling_and_a_cut_for_a_fade():
    """The window's knobs travel in the list, so a saved workflow sings what was asked for."""
    edits = track.read('[{"op": "retake", "bars": [0, 4], "vary": 1.6, "guide": 2.5},'
                       ' {"op": "cut", "bars": [4, 8], "fade": 1.5}]')
    assert (edits[0].vary, edits[0].guide, edits[0].fade) == (1.6, 2.5, None)
    assert (edits[1].vary, edits[1].guide, edits[1].fade) == (None, None, 1.5)
    assert track.read(track.written(edits)) == edits, "and they come back through the field"
    plain = track.read('[{"op": "retake", "bars": [0, 4]}, {"op": "cut", "bars": [4, 8]}]')
    assert [edit.vary for edit in plain] == [None, None], "an edit that says nothing asks none"
    assert "vary" not in track.written(plain), "and nothing of it is written down"


@pytest.mark.parametrize("item, says", [
    ('{"op": "retake", "bars": [0, 4], "vary": 9}', "the variety is 9"),
    ('{"op": "retake", "bars": [0, 4], "guide": 0.5}', "the guide is 0.5"),
    ('{"op": "retake", "bars": [0, 4], "vary": "warm"}', "the variety is not a number"),
    ('{"op": "cut", "bars": [0, 4], "fade": 30}', "the fade is 30"),
])
def test_a_knob_outside_what_it_can_be_is_refused(item, says):
    with pytest.raises(ValueError, match=says):
        track.read("[" + item + "]")


def test_the_takes_of_an_edit_follow_its_sampling_and_not_its_fade():
    """Another temperature is another take; a longer fade is the same sound with a ramp on it."""
    one, other = track.read('[{"op": "retake", "bars": [0, 4], "seed": 3},'
                            ' {"op": "retake", "bars": [0, 4], "seed": 3, "vary": 1.5}]')
    assert track.name("song", [], one) != track.name("song", [], other)
    guided = track.read('[{"op": "retake", "bars": [0, 4], "seed": 3, "guide": 3.0}]')[0]
    assert track.name("song", [], guided) != track.name("song", [], one)
    plain, faded = track.read('[{"op": "cut", "bars": [0, 4]},'
                              ' {"op": "cut", "bars": [0, 4], "fade": 2.0}]')
    assert track.name("song", [], plain) == track.name("song", [], faded)


def test_the_name_of_a_result_holds_the_song_the_edits_and_the_takes_kept():
    edits = track.read('[{"op": "cut", "bars": [2, 4]}, {"op": "retake", "bars": [2, 6], "seed": 5}]')
    cut, retake = edits
    first = track.name("song", [], cut)
    assert first == track.name("song", [], cut)
    assert first != track.name("other", [], cut)
    assert first != track.name("song", [(cut, 0)], cut)
    after = track.name("song", [(cut, 0)], retake)
    assert after != track.name("song", [(cut, 1)], retake)
    assert len(first) == 64


def test_asking_for_another_take_does_not_rename_what_is_already_sung():
    """More takes sings the one that is missing; the ones already there keep their name."""
    two = track.read('[{"op": "retake", "bars": [2, 6], "seed": 5, "takes": 2}]')[0]
    three = track.read('[{"op": "retake", "bars": [2, 6], "seed": 5, "takes": 3, "take": 2}]')[0]
    assert track.name("song", [], two) == track.name("song", [], three)
    assert three.seeds()[:2] == two.seeds()


TWICE = score(("intro", ["z16", "z8z2DDA2AA"]),
              ("verse", ["A2DDAAA2AGGGG2GG", "AGGGGFFFFDD2B4"]),
              ("chorus", ["A2z8D2B4", "A2z4FG3F2B2B2"]),
              ("verse", ["A2DDAAA2AGGGG2GG", "AGGGGFFFFDD2B4"]),
              ("chorus", ["A2z8D2B4", "A2z4FG3F2B2B2"]),
              ("outro", ["z16", "z16"]))
"""Two choruses: intro 0-1, verse 2-3, chorus 4-5, verse 6-7, chorus 8-9, outro 10-11."""

TWICE_WORDS = ("[Verse]\none two three\nfour five six\n\n[Chorus]\nseven eight\n\n"
               "[Verse]\nnine ten\n\n[Chorus]\nseven eight")

TWICE_FRAMES = 640
"""The twelve bars from 0.5 s, two seconds each, and a little after them."""

WRITTEN_ON = ("V: Vocal\nA2z8D2B4|A2z4FG3F2B2B2|A2z8DDBB3|\nV: Ins\nZ|Z|Z|\n"
              "% verse\nV: Vocal\nA2DDAAA2AGGGG2GG|AGGGGFFFFDD2B4|\nV: Ins\nZ|Z|\n"
              "% chorus\nV: Vocal\nA2z8D2B4|A2z4FG3F2B2B2|A2z8DDBB3|\nV: Ins\nZ|Z|Z|\n"
              "% outro\nV: Vocal\nz16|z16|\nV: Ins\nZ|Z|\n")
"""What a stand-in for the model writes after the first verse: three-bar choruses."""


def twice_state(offset=0.5, lyrics=TWICE_WORDS):
    sheet = notation.read(TWICE)
    clock = grid.Grid(offset=offset, rate=1.0, tick=grid.tick_seconds(sheet),
                      starts=grid.starts_of(sheet))
    song = types.SimpleNamespace(score=TWICE, lyrics=lyrics, frames=TWICE_FRAMES)
    return track.opened(song, clock)


def restyle(**fields):
    item = {"op": "restyle", "bars": [4, 6], "style": "hip hop, male rapper", "seed": 3}
    item.update(fields)
    return track.read(json.dumps([item]))[0]


def written_twice():
    """A new melody for both choruses: the plan, the score the model wrote and its grid."""
    state = twice_state()
    step = track.plan(state, restyle(melody="new", also=[[8, 10]]))
    written = step.score + WRITTEN_ON
    sheet = notation.read(written)
    other = grid.Grid(offset=0.3, rate=1.0, tick=grid.tick_seconds(sheet),
                      starts=grid.starts_of(sheet))
    return state, step, written, sheet, other


def test_a_restyle_reads_as_the_window_writes_it_and_writes_back_the_same():
    edit = restyle(also=[[8, 10]], melody="new", octave=-1, takes=2, take=1, vary=1.2)
    assert (edit.style, edit.melody, edit.octave, edit.also) == (
        "hip hop, male rapper", "new", -1, ((8, 10),))
    assert (edit.bars, edit.seed, edit.takes, edit.take, edit.vary) == ((4, 6), 3, 2, 1, 1.2)
    plain = restyle(style="  soft folk \n")
    assert (plain.style, plain.melody, plain.octave, plain.also) == ("soft folk", "same", 0, ())
    assert json.loads(track.written([plain]))[0] == {
        "op": "restyle", "bars": [4, 6], "style": "soft folk", "melody": "same", "octave": 0,
        "seed": 3, "takes": 1}
    assert track.read(track.written([edit])) == (edit,)


@pytest.mark.parametrize("fields,says", [
    ({"bars": None, "seconds": [1.0, 4.0]}, "it selects bars, not seconds"),
    ({"style": "  "}, "gives no style"),
    ({"style": None}, "gives no style"),
    ({"style": "x" * 1001}, "1001 characters"),
    ({"melody": "rap"}, "the melody 'rap'"),
    ({"octave": 2}, "moves the voice 2 octaves"),
    ({"octave": 0.5}, "the octave is not a whole number"),
    ({"also": [[5, 9]]}, "overlap"),
    ({"also": [[8]]}, "not a pair"),
    ({"also": {"bars": 1}}, "not a list"),
    ({"also": [[9, 8]]}, "not a stretch"),
])
def test_a_restyle_that_cannot_be_sung_is_refused_in_words(fields, says):
    item = {"op": "restyle", "bars": [4, 6], "style": "hip hop", "seed": 3}
    item.update(fields)
    if item["bars"] is None:
        del item["bars"]
    with pytest.raises(ValueError, match=says):
        track.read(json.dumps([item]))


def test_the_takes_of_a_restyle_are_named_by_its_style_its_melody_and_its_voice():
    first = restyle()
    names = {track.name("song", [], edit) for edit in (
        first, restyle(style="soft folk"), restyle(melody="new"), restyle(octave=1),
        restyle(also=[[8, 10]]))}
    assert len(names) == 5
    assert track.name("song", [], first) == track.name("song", [], restyle(takes=3, take=2))


def test_a_restyle_of_the_same_notes_sings_the_song_s_own_score_over_any_bars():
    state = twice_state()
    step = track.plan(state, restyle(bars=[5, 6], also=[[8, 10]]))
    sheet, clock = state.sheet, state.clock
    marks = grid.seams(sheet)
    beat, margin = grid._beat_and_margin(sheet, clock)
    opens = clock.moment(4, grid.opening(marks[4], beat, margin))
    retake = grid.retake_frames(sheet, clock, marks, 5, 6, state.frames)[0]
    assert (step.kind, step.stretches, step.blocks, step.score) == (
        "restyle", ((5, 6), (8, 10)), (), TWICE)
    assert step.start == min(opens, retake - track.SUNG_LEAST)
    assert step.stop == grid.retake_frames(sheet, clock, marks, 8, 10, state.frames)[1]
    assert step.lyrics == TWICE_WORDS and step.bars == (5, 10)
    assert track.restyle_spans(state, step, TWICE) == ((5, 6), (8, 10))
    bars, seconds = track.restyle_reach(state, step, sheet, ((5, 6), (8, 10)))
    assert bars == 12 and seconds == pytest.approx(0.5 + 24.0 + track.RESTYLE_TAIL)
    with pytest.raises(ValueError, match="has 10 bars where the song has 12"):
        track.restyle_spans(state, step, notation.head(TWICE, 10))


def test_a_restyle_reaching_the_end_of_the_song_sings_the_other_song_whole():
    state = twice_state()
    step = track.plan(state, restyle(bars=[10, 12]))
    assert track.restyle_reach(state, step, state.sheet, ((10, 12),)) == (12, None)


@pytest.mark.parametrize("fields,says", [
    ({"bars": [4, 5]}, "bar line 6 is inside the chorus"),
    ({"bars": [0, 2]}, "the first section has none"),
    ({"bars": [8, 12]}, "the outro at bar 11 sings none of them"),
])
def test_a_new_melody_goes_by_whole_sections_that_sing(fields, says):
    with pytest.raises(ValueError, match=says):
        track.plan(twice_state(), restyle(melody="new", **fields))


def test_a_new_melody_needs_the_words_matched_with_the_sections():
    state = twice_state(lyrics="[Chorus]\none\n\n[Chorus]\ntwo")
    with pytest.raises(ValueError, match="could not be matched"):
        track.plan(state, restyle(melody="new"))


def test_a_new_melody_is_written_before_the_sections_part_from_the_order_of_the_words():
    """The model sang a bridge where the words have their second verse: the first chorus is still told."""
    parted = "[Verse]\none two three\n\n[Chorus]\nseven eight\n\n[Bridge]\nnine ten\n\n[Chorus]\nseven"
    assert ops.sung_sections(parted, TWICE) == [1, 2, None, None]
    state = twice_state(lyrics=parted)
    assert track.plan(state, restyle(melody="new")).blocks == ((1, 1),)
    with pytest.raises(ValueError, match="from the verse at bar 7 on"):
        track.plan(state, restyle(melody="new", also=[[8, 10]]))
    split = "[Verse]\none\n\n[Verse]\ntwo\n\n[Verse]\nthree"
    assert ops.sung_sections(split, TWICE) == [1, None, None], "a sung chorus is not stepped over"
    assert ops.sung_sections(TWICE_WORDS, TWICE) == [1, 2, 3, 4], "the intro's notes are"


def test_a_new_melody_is_written_on_from_the_bars_before_it_and_found_again_by_its_words():
    state, step, written, sheet, _other = written_twice()
    assert step.blocks == ((1, 1), (3, 3)), "the first chorus and the second, among the sung blocks"
    assert step.score.endswith("% chorus\n") and notation.head_bars(step.score) == 4
    assert track.restyle_spans(state, step, written) == ((4, 7), (9, 12))
    bars, seconds = track.restyle_reach(state, step, sheet, ((4, 7), (9, 12)))
    assert bars == 14 and seconds == pytest.approx(0.5 + 28.0 + track.RESTYLE_TAIL)
    renamed = step.score + WRITTEN_ON.replace("% verse", "% bridge")
    with pytest.raises(ValueError, match="could not be matched"):
        track.restyle_spans(state, step, renamed)


def test_a_restyle_lays_both_songs_cut_at_one_shift_and_sings_the_bar_before_each_seam():
    state, step, written, sheet, other = written_twice()
    spans = track.restyle_spans(state, step, written)
    laid = track.restyle_layout(state, step, sheet, other, 720, spans)
    assert [piece[0] for piece in laid.pieces] == [0, 1, 0, 1, 0]
    assert laid.stretches == ((0, 0, 4), (1, 4, 7), (0, 6, 8), (1, 9, 12), (0, 10, 12))
    marks, theirs = grid.seams(state.sheet), grid.seams(sheet)
    beat, margin = grid._beat_and_margin(state.sheet, state.clock)
    enter = grid.shared_shift([marks[4], theirs[4]], beat, margin)
    leave = grid.shared_shift([marks[6], theirs[7]], beat, margin)
    assert laid.pieces[0] == (0, 0, state.clock.moment(4, enter))
    assert laid.pieces[1] == (1, other.moment(4, enter), other.moment(7, leave))
    assert laid.pieces[2][1] == state.clock.moment(6, leave)
    assert laid.pieces[-1] == (0, laid.pieces[-1][1], TWICE_FRAMES)
    assert laid.owners == (1, 0, 1, 0), "into the new style under its prompt, out under the song's"
    seams, at = [], 0
    for _which, start, stop in laid.pieces[:-1]:
        at += stop - start
        seams.append(at)
    assert [seam for _start, seam in laid.sung] == seams
    assert all(seam - start >= track.SUNG_LEAST for start, seam in laid.sung)
    assert laid.sung[0][0] == step.start, "the bar before the first stretch, where the plan opens"
    assert laid.lines == other.bar_seconds()
    with pytest.raises(ValueError, match="stops before the bars"):
        track.restyle_layout(state, step, sheet, other, 500, spans)


def test_the_song_after_a_restyle_has_the_other_song_s_bars_where_they_were_laid():
    state, step, written, sheet, other = written_twice()
    spans = track.restyle_spans(state, step, written)
    laid = track.restyle_layout(state, step, sheet, other, 720, spans)
    score = notation.restyled(TWICE, written, [(4, 6, 4, 7), (8, 10, 9, 12)])
    left = track.after(state, step, 0, score, (), laid)
    assert len(left.sheet["bars"]) == 14 and left.lyrics == TWICE_WORDS
    assert [section["name"] for section in left.sheet["sections"]] == [
        "intro", "verse", "chorus", "verse", "chorus", "outro"]
    assert left.frames == sum(stop - start for _which, start, stop in laid.pieces)
    (_own, _zero, cut), (_donor, begin, end) = laid.pieces[:2]
    assert left.clock.at(5) == pytest.approx(other.at(5) + (cut - begin) * 0.04)
    assert left.clock.at(7) == pytest.approx(
        state.clock.at(6) + (cut + end - begin - laid.pieces[2][1]) * 0.04)
    sung = ((laid.sung[0][0], laid.sung[0][1], laid.sung[0][1] - laid.sung[0][0] + 5),)
    longer = track.after(state, step, 0, score, sung, laid)
    assert longer.frames == left.frames + 5
    assert longer.clock.at(13) == pytest.approx(left.clock.at(13) + 0.2)
