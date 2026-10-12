"""The words of a song laid under the notes of its score by order, and the route the piano roll asks."""

from __future__ import annotations

import json
import pathlib
import re

from yue2_comfy import notation, routes, underlay

SONGS = json.loads(
    (pathlib.Path(__file__).resolve().parent / "data" / "underlay_songs.json").read_text(encoding="utf-8"))
"""Two songs of the measurement of 2026-10-12: the score YuE2 wrote for each, and the four sets of words sung over it.

``own`` are the words the score was written for; ``same`` other words with as
many syllables; ``more`` and ``less`` the verses with two syllables a line
more or fewer in the first and one in the second. The choruses are the same in
all four.
"""

SEVEN = "C2D2E2F2G2A2B2z2|"
EIGHT = "C2D2E2F2G2A2B1c1z2|"
THREE = "c4B4A4z4|"
REST = "z16|"

VERSE = "[Verse]\nla la la la la la la\nna na na na na na na\n"
CHORUS = "[Chorus]\noh oh oh\nhey hey hey\n"


def bars_in(line: str) -> int:
    count = 0
    for bar in line[:-1].split("|"):
        found = re.fullmatch(r"Z([2-4])?", bar.strip())
        count += int(found.group(1) or "1") if found else 1
    return count


def score(*groups) -> str:
    """A score in YuE2's dialect: each group is ``(section name or None, the voice's bars)``, the instrument silent."""
    text = ["X:1", "T:", "M:4/4", "L:1/16", "Q:1/4=100",
            'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
            'V: Ins clef=treble name="Ins Melody" snm="Inst."', "K:C"]
    for name, vocal in groups:
        count = bars_in(vocal)
        if name:
            text.append("% " + name)
        text.extend(["V: Vocal", vocal, "V: Ins", "Z{}|".format(count) if count > 1 else "Z|"])
    return "\n".join(text) + "\n"


def lay(text: str, lyrics: str) -> dict:
    return underlay.lay(notation.read(text), lyrics)


def spans(laid: dict) -> list:
    return [(row["first"], row["stop"]) for row in laid["lines"]]


def kinds(laid: dict) -> str:
    return "".join(row["kind"][0] for row in laid["lines"])


def texts(laid: dict) -> list:
    return [text for _note, _line, text in laid["notes"]]


SONG = score(("intro", REST), ("verse", SEVEN + SEVEN), ("chorus", THREE + THREE))


def test_a_line_with_a_note_for_every_syllable_is_laid_a_syllable_a_note():
    laid = lay(SONG, VERSE + CHORUS)
    assert laid["paired"] and not laid["bare"]
    assert spans(laid) == [(0, 7), (7, 14), (14, 17), (17, 20)]
    assert kinds(laid) == "ffff"
    assert laid["counts"] == {"fit": 4, "spare": 0, "crowd": 0}
    assert laid["notes"][:8] == [[0, 0, "la"], [1, 0, "la"], [2, 0, "la"], [3, 0, "la"], [4, 0, "la"],
                                 [5, 0, "la"], [6, 0, "la"], [7, 1, "na"]]
    assert texts(laid)[14:] == ["oh", "oh", "oh", "hey", "hey", "hey"]
    assert laid["spare"] == []
    assert laid["sung"] == 20


def test_lines_are_numbered_as_the_lyrics_are_written():
    laid = lay(SONG, "\n" + VERSE + "\n" + CHORUS)
    assert [row["line"] for row in laid["lines"]] == [2, 3, 6, 7]
    assert laid["lines"][0]["text"] == "la la la la la la la"
    assert laid["lines"][0]["syllables"] == 7 and laid["lines"][0]["notes"] == 7


def test_a_word_of_several_syllables_is_hyphened_across_its_notes():
    laid = lay(score(("verse", EIGHT)), "[Verse]\nMorning breaks above the river\n")
    assert texts(laid) == ["Mor-", "ning", "breaks", "a-", "bove", "the", "ri-", "ver"]
    assert kinds(laid) == "f"


def test_syllables_left_over_share_the_last_note_of_their_line():
    laid = lay(SONG, "[Verse]\nla la la la la la morning light\nna na na na na na na\n" + CHORUS)
    assert kinds(laid) == "cfff"
    assert spans(laid)[:2] == [(0, 7), (7, 14)]
    assert laid["notes"][5:8] == [[5, 0, "la"], [6, 0, "morning light"], [7, 1, "na"]]
    assert laid["lines"][0]["syllables"] == 9 and laid["lines"][0]["notes"] == 7


def test_a_word_cut_by_the_last_note_is_written_whole_under_it():
    laid = lay(score(("verse", THREE)), "[Verse]\nsing tomorrow\n")
    assert texts(laid) == ["sing", "to-", "morrow"]
    assert kinds(laid) == "c"


def test_notes_left_over_are_listed_and_the_next_line_starts_on_its_own_phrase():
    laid = lay(SONG, "[Verse]\nla la la la la\nna na na na na na na\n" + CHORUS)
    assert kinds(laid) == "sfff"
    assert spans(laid)[:2] == [(0, 7), (7, 14)]
    assert laid["spare"] == [5, 6]
    assert [note for note, line, _text in laid["notes"] if line == 0] == [0, 1, 2, 3, 4]


def test_a_russian_word_with_no_vowel_is_written_with_the_word_after_it():
    assert [part[0] for part in underlay.parts_of("\u0432 \u0434\u043e\u043c\u0435 \u0442\u0438\u0445\u043e")] == [
        "\u0432 \u0434\u043e", "\u043c\u0435", "\u0442\u0438", "\u0445\u043e"]
    assert [part[0] for part in underlay.parts_of("\u043d\u0443 \u0438 \u0436")] == ["\u043d\u0443", "\u0438 \u0436"]
    laid = lay(score(("verse", "C4D4E4F4|")), "[Verse]\n\u0432 \u0434\u043e\u043c\u0435 \u0442\u0438\u0445\u043e\n")
    assert texts(laid) == ["\u0432 \u0434\u043e-", "\u043c\u0435", "\u0442\u0438-", "\u0445\u043e"]


def test_a_line_with_no_word_in_it_is_not_laid():
    blocks = underlay.blocks_of("[Verse]\n...\nla la\n(1, 2, 3)\n[Bridge]\n\n[Chorus]\noh\n")
    assert [(label, [row["line"] for row in rows]) for label, rows in blocks] == [("verse", [2]), ("chorus", [7])]
    assert underlay.blocks_of("la la\n[Chorus]\noh\n")[0][0] == "verse"


def test_sections_are_paired_by_name_past_an_intro_the_voice_hums_in():
    hummed = score(("intro", "c4c4z8|"), ("verse", SEVEN + SEVEN), ("interlude", "c8z8|"),
                   ("chorus", THREE + THREE))
    laid = lay(hummed, VERSE + CHORUS)
    assert laid["paired"]
    assert spans(laid) == [(2, 9), (9, 16), (17, 20), (20, 23)]
    assert kinds(laid) == "ffff"


def test_a_line_sung_from_the_bar_before_its_section_belongs_to_that_section():
    picked = score(("intro", "z12C2D2|"), ("verse", "E2F2G2A2B2z6|"), ("chorus", THREE))
    laid = lay(picked, "[Verse]\nla la la la la la la\n[Chorus]\noh oh oh\n")
    assert spans(laid) == [(0, 7), (7, 10)]
    assert kinds(laid) == "ff"


def test_sections_named_in_another_order_are_paired_by_their_sizes():
    swapped = score(("intro", "c4c4z8|"), ("chorus", SEVEN + SEVEN), ("verse", THREE + THREE))
    laid = lay(swapped, VERSE + CHORUS)
    assert laid["paired"]
    assert spans(laid) == [(2, 9), (9, 16), (16, 19), (19, 22)]
    assert kinds(laid) == "ffff"


def test_pairing_by_name_and_by_size():
    sections = [("intro", 2), ("verse", 14), ("chorus", 6), ("verse", 14), ("chorus", 6), ("outro", 0)]
    labels = ["verse", "chorus", "verse", "chorus"]
    assert underlay.by_name(labels, sections) == [1, 2, 3, 4]
    assert underlay.by_size(labels, [14, 6, 14, 6], sections) == [1, 2, 3, 4]
    assert underlay.by_name(["verse", "bridge"], sections) is None
    assert underlay.by_size(["verse", "bridge"], [14, 6], sections) == [1, 2]
    assert underlay.by_name(["verse", "verse"], [("verse", 7), ("chorus", 7)]) == [0, 1]
    assert underlay.by_size(["verse", "chorus", "bridge"], [7, 7, 7], [("verse", 7), ("chorus", 0)]) is None


def test_lyrics_with_more_sections_than_the_voice_sings_in_are_laid_along_the_whole_song():
    laid = lay(score(("verse", SEVEN + SEVEN)), VERSE + CHORUS + "[Bridge]\nla la\n")
    assert not laid["paired"]
    assert len(laid["lines"]) == 5
    assert laid["lines"][0]["first"] == 0 and laid["lines"][-1]["stop"] == 14


def test_a_line_no_note_is_left_for_is_crowded_and_draws_nothing():
    laid = lay(score(("verse", THREE)), "[Verse]\noh\nhey\nla\nna\nda\n")
    assert [row["notes"] for row in laid["lines"]] == [1, 1, 1, 0, 0]
    assert kinds(laid) == "fffcc"
    assert laid["notes"] == [[0, 0, "oh"], [1, 1, "hey"], [2, 2, "la"]]


def test_a_score_that_names_no_section_is_not_laid():
    laid = lay(score((None, SEVEN + SEVEN)), VERSE)
    assert laid["bare"]
    assert laid["lines"] == [] and laid["notes"] == []


def test_no_words_lay_nothing():
    for lyrics in ("", "[Verse]\n\n[Chorus]\n", "...\n"):
        laid = lay(SONG, lyrics)
        assert not laid["bare"]
        assert laid["lines"] == [] and laid["notes"] == [] and laid["spare"] == []


def test_a_score_with_no_voice_leaves_every_line_without_a_note():
    laid = lay(score(("verse", REST + REST)), VERSE)
    assert laid["sung"] == 0
    assert kinds(laid) == "cc" and laid["notes"] == []


def test_the_models_own_words_fit_the_score_it_wrote_for_them():
    """Three lines in four have a note for every syllable; the fourth of a section takes the notes left over."""
    for name, counts in (("en_pop", {"fit": 12, "spare": 4, "crowd": 0}),
                         ("ru_pop", {"fit": 13, "spare": 3, "crowd": 0})):
        laid = lay(SONGS[name]["score"], SONGS[name]["lyrics"]["own"])
        assert laid["paired"]
        assert laid["counts"] == counts
        assert len(laid["lines"]) == 16
        starts = [row["first"] for row in laid["lines"]]
        assert starts == sorted(starts)


def test_rewritten_words_are_laid_on_the_notes_the_lines_had():
    """The cuts follow the rests of the tune, so a line longer or shorter than its phrase is marked where it is."""
    for name in SONGS:
        own = lay(SONGS[name]["score"], SONGS[name]["lyrics"]["own"])
        for kind in ("same", "more", "less"):
            laid = lay(SONGS[name]["score"], SONGS[name]["lyrics"][kind])
            assert spans(laid) == spans(own), (name, kind)
        assert kinds(lay(SONGS[name]["score"], SONGS[name]["lyrics"]["same"])) == kinds(own)
    assert kinds(lay(SONGS["en_pop"]["score"], SONGS["en_pop"]["lyrics"]["more"])) == "ccccfffscccffffs"
    assert kinds(lay(SONGS["en_pop"]["score"], SONGS["en_pop"]["lyrics"]["less"])) == "ssssfffsssssfffs"
    assert kinds(lay(SONGS["ru_pop"]["score"], SONGS["ru_pop"]["lyrics"]["more"])) == "ccccfsffccccfssf"
    assert kinds(lay(SONGS["ru_pop"]["score"], SONGS["ru_pop"]["lyrics"]["less"])) == "ssssfsffssssfssf"


def test_every_syllable_of_a_song_is_under_a_note_in_order():
    for name in SONGS:
        for kind, lyrics in SONGS[name]["lyrics"].items():
            laid = lay(SONGS[name]["score"], lyrics)
            notes = [note for note, _line, _text in laid["notes"]]
            assert notes == sorted(set(notes)), (name, kind)
            assert not set(notes) & set(laid["spare"])
            drawn = "".join(text.rstrip("-") for _note, _line, text in laid["notes"]).replace(" ", "")
            written = "".join("".join(part[0] for part in row["parts"]).replace(" ", "")
                              for _label, rows in underlay.blocks_of(lyrics) for row in rows)
            assert drawn == written, (name, kind)


def test_the_words_route_lays_the_lyrics_under_the_score():
    payload, status = routes.answer_score_words({"abc": SONG, "lyrics": VERSE + CHORUS})
    assert status == 200 and payload["ok"]
    assert payload["counts"] == {"fit": 4, "spare": 0, "crowd": 0}
    assert payload["notes"][0] == [0, 0, "la"]
    assert json.loads(json.dumps(payload)) == payload


def test_the_words_route_refuses_what_the_editor_never_sends():
    for body in (None, [], {"abc": SONG}, {"lyrics": "la"}, {"abc": 5, "lyrics": "la"}, {"abc": SONG, "lyrics": 5}):
        assert routes.answer_score_words(body)[1] == 400
    payload, status = routes.answer_score_words({"abc": SONG, "lyrics": "la " * routes.LONGEST})
    assert status == 413 and not payload["ok"]


def test_the_words_route_says_why_a_score_cannot_be_laid():
    payload, status = routes.answer_score_words({"abc": "junk", "lyrics": VERSE})
    assert status == 200 and not payload["ok"] and payload["error"]
    payload, status = routes.answer_score_words({"abc": score((None, SEVEN)), "lyrics": VERSE})
    assert status == 200 and payload["ok"] and payload["bare"]
