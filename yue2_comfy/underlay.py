"""The words of a song under the notes of its score, laid by order: which syllable sits on which note.

A score has notes and no words. Which syllable is sung on which note is
decided by the model while it sings, so a picture of the words under the notes
can only be drawn two ways: by order, before anything is sung, or from what
was heard afterwards. Measured on 2026-10-12 on 28 songs -- 4 the pack had
kept and 24 sung for the measurement, six songs under the score the model
wrote for them and that score again at the same seed under verses rewritten
three ways: other words with as many syllables, one or two syllables more a
line, one or two fewer. The forced aligner heard the separated voice a section
at a time, and a word was counted only when a second hearing, of the song as
mixed, put it on the same note:

- YuE2 writes a note for every syllable. Under its own words 154 lines of 184
  had exactly as many notes as syllables, each line on a phrase of its own.
- So the words are laid by order: the sections of the lyrics on the sections
  of the score, a section's lines cut where the tune rests, a line's syllables
  on its notes one by one. On lines with as many notes as syllables, 92 words
  in 100 were heard on the note they are laid on in the songs sung for this,
  84 in the kept ones and 91 under rewritten words of the same length; within
  one note, 99 in all three.
- Where a line has more syllables than notes, 65 percent, 85 within one note;
  where it has fewer, 57 and 85. There the model sings its own way: syllables
  left over run past the phrase, and with too few the next line starts early
  and the words run ahead, by 3 to 9 notes in one song. That cannot be told to
  the note, so such a line is marked instead of guessed at.
- The cuts hold when the words change: under every rewriting all 96 lines lay
  on the notes they had under the model's own words, 46 of the 48 lengthened
  lines came out marked as crowded and 46 of the 48 shortened ones as having
  notes to spare.
- What lost: lines cut by their syllable counts alone, with no eye for the
  rests, 50 to 53 percent once the counts had changed; syllables left over
  spread evenly through the line, 23 percent against 65 on its last note; the
  whole song laid as one stretch with no sections, 80 percent against 85.
- A picture from what was heard would not be better: two hearings of one song
  put 71 to 78 percent of its words on the same note.

Sections are paired by name first, each lyrics section taking the next score
section named as its tag reads, as an edit of the track pairs them. In one
song of ten the names did not line up -- the model had named its sections in
another order -- and there the sections are paired by size, each lyrics
section on a score section with about as many notes as it has syllables;
where the names do line up, sizes give the same pairs.

Not measured: scores written from a recording, rap, and languages other than
Russian and English. A score that names no section is not laid at all: such a
tune is laid out for its words when it is sung (see ``phrasing``), on bars of
the node's choosing, so its notes in order say nothing about its words.

Standard library and the pack's own readers; no torch.
"""

from __future__ import annotations

import math

from . import phrasing, syllables
from .inpaint import grid

FIT = "fit"
"""A line with as many notes as syllables: the one the measurement found sung where it is laid."""

SPARE = "spare"
"""A line with notes left over: its last notes are drawn with no syllable under them."""

CROWD = "crowd"
"""A line with syllables left over, drawn together under its last note; also a line no note was left for."""

MID = 0.6
"""What it costs to end a line where the tune does not rest for an eighth note."""

LONGEST = 40
"""Notes one line may be laid on."""

UNPLACED = 3.0
"""What a line costs that no note is left for."""

OTHER_NAME = 0.5
"""What it costs a lyrics section, paired by size, to be laid on a score section of another name."""

PASSED = 0.04
"""What each sung note of a score section costs when no lyrics section is laid on it."""

LATER = 0.001
"""What each section further into the song costs a pairing by size, so of two as good the earlier is taken."""


def parts_of(line: str) -> list:
    """The syllables of a line of lyrics as ``[text, word, closes]``, in order.

    ``word`` counts the line's words from 0 and ``closes`` says the syllable
    is the last of its word. A word that is no syllable of its own -- a
    Russian preposition of one consonant -- is written before the first
    syllable of the word after it, or after the last syllable of the line.
    """
    found = []
    waiting = ""
    for index, word in enumerate(syllables.words(line)):
        cut = syllables.pieces(word)
        if not cut:
            waiting = (waiting + " " + word).strip()
            continue
        for place, piece in enumerate(cut):
            if place == 0 and waiting:
                piece = waiting + " " + piece
                waiting = ""
            found.append([piece, index, place == len(cut) - 1])
    if waiting and found:
        found[-1][0] += " " + waiting
    elif waiting:
        found.append([waiting, 0, True])
    return found


def blocks_of(lyrics: str) -> list:
    """The lyrics as ``[label, [row]]`` sections in order, a section for every tag with a sung line under it.

    A row is ``{"line", "text", "parts"}``: the line's number counted from 0
    over the lyrics as they are written, tags and blank lines included, the
    line itself, and its syllables from ``parts_of``. Lines before the first
    tag are a verse, as ``phrasing.lyric_sections`` reads them.
    """
    found = []
    current = None
    for number, raw in enumerate(str(lyrics or "").splitlines()):
        line = raw.strip()
        if not line:
            continue
        tag = phrasing.TAG.fullmatch(line)
        if tag:
            current = [phrasing.label_of(tag.group(1)), []]
            found.append(current)
            continue
        parts = parts_of(line)
        if not parts:
            continue
        if current is None:
            current = ["verse", []]
            found.append(current)
        current[1].append({"line": number, "text": line, "parts": parts})
    return [block for block in found if block[1]]


def by_name(labels, sections):
    """The score section each lyrics section is laid on, by name, or None when the names do not line up.

    The rule ``inpaint.ops`` pairs by when a cut or a move takes words along
    with their bars: each label takes the next section of its name, which
    steps over intros and interludes, and when that fails and the voice sings
    in exactly as many sections as the lyrics have, those in order.
    ``sections`` is ``(name, notes)`` for each section of the score.
    """
    found = []
    at = 0
    for label in labels:
        while at < len(sections) and sections[at][0] != label:
            at += 1
        if at == len(sections):
            break
        found.append(at)
        at += 1
    if len(found) == len(labels):
        return found
    sung = [index for index, section in enumerate(sections) if section[1]]
    return sung if len(sung) == len(labels) else None


def by_size(labels, counts, sections):
    """The score section each lyrics section is laid on, in order, by how their sizes agree; None when there are too few.

    ``counts`` holds the syllables of each lyrics section. A lyrics section
    wants a score section with about as many notes as it has syllables and,
    if it can have it, its own name; a score section with notes in it that is
    passed over costs by those notes.
    """
    blocks, size = len(labels), len(sections)
    infinity = float("inf")
    best = [[infinity] * (size + 1) for _ in range(blocks + 1)]
    back = [[None] * (size + 1) for _ in range(blocks + 1)]
    best[0][0] = 0.0
    for block in range(blocks + 1):
        for at in range(size):
            here = best[block][at]
            if here == infinity:
                continue
            name, notes = sections[at]
            if here + PASSED * notes < best[block][at + 1]:
                best[block][at + 1] = here + PASSED * notes
                back[block][at + 1] = (block, at, False)
            if block < blocks and notes:
                value = here + abs(math.log((notes + 1.0) / (counts[block] + 1.0)))
                value += (0.0 if name == labels[block] else OTHER_NAME) + LATER * at
                if value < best[block + 1][at + 1]:
                    best[block + 1][at + 1] = value
                    back[block + 1][at + 1] = (block, at, True)
    if best[blocks][size] == infinity:
        return None
    found = []
    block, at = blocks, size
    while (block, at) != (0, 0):
        block, at, took = back[block][at]
        if took:
            found.append(at)
    found.reverse()
    return found


def _sections(sheet, onsets) -> list:
    """``(name, notes)`` for each section of a sheet: its name as a lyrics tag reads, and the vocal notes that start in its bars."""
    bars = sheet["bars"]
    found = []
    for section in sheet["sections"]:
        first, last = section["bar"], section["bar"] + section["bars"] - 1
        low, high = bars[first]["start"], bars[last]["start"] + bars[last]["length"]
        found.append((" ".join(str(section["name"]).lower().split()),
                      sum(1 for tick in onsets if low <= tick < high)))
    return found


def stretches(sheet, notes, blocks) -> tuple:
    """``([(rows, first note, stop note)], paired)``: the run of notes each lyrics section is laid on.

    A section's notes are the ones from where its singing opens to where the
    next section's does: a line sung from the bar before its section's first,
    a pickup, belongs to the section it leads into (``inpaint.grid.seams``).
    When the sections cannot be paired, every line is laid along the whole
    song as one stretch and ``paired`` is False.
    """
    onsets = [note["start"] for note in notes]
    labels = [label for label, _rows in blocks]
    counts = [sum(len(row["parts"]) for row in rows) for _label, rows in blocks]
    sections = _sections(sheet, onsets)
    pairs = by_name(labels, sections)
    if pairs is None:
        pairs = by_size(labels, counts, sections)
    if pairs is None:
        return [([row for _label, rows in blocks for row in rows], 0, len(notes))], False
    marks = grid.seams(sheet)
    starts = grid.starts_of(sheet)

    def opening(bar):
        return starts[bar] - max(0.0, marks[bar].pickup)

    found = []
    for (_label, rows), at in zip(blocks, pairs):
        section = sheet["sections"][at]
        low, high = opening(section["bar"]), opening(section["bar"] + section["bars"])
        inside = [index for index, tick in enumerate(onsets) if low <= tick < high]
        found.append((rows, inside[0] if inside else 0, inside[-1] + 1 if inside else 0))
    return found, True


def cut(notes, counts, first: int, stop: int, per_quarter) -> list:
    """``(first note, stop note)`` for each line: notes ``first`` to ``stop`` cut into one run a line, in order.

    The cheapest cut: a line wants as many notes as it has syllables
    (``counts``), and a cut wants a rest of an eighth note or more after the
    note it ends on. The last line takes the notes to the end of the stretch,
    and a line no note is left for is an empty run.
    """
    size = stop - first
    lines = len(counts)
    if size <= 0:
        return [(first, first)] * lines
    breath = per_quarter / 2.0
    rest = [0.0] * (size + 1)
    for index in range(1, size):
        before, after = notes[first + index - 1], notes[first + index]
        rest[index] = 0.0 if after["start"] - (before["start"] + before["length"]) >= breath else MID
    logs = [0.0] + [math.log(take) for take in range(1, LONGEST + 1)]
    infinity = float("inf")
    best = [[infinity] * (size + 1) for _ in range(lines + 1)]
    back = [[0] * (size + 1) for _ in range(lines + 1)]
    best[0][0] = 0.0
    for line in range(lines):
        want = math.log(max(1, counts[line]))
        last = line == lines - 1
        row, following, origin = best[line], best[line + 1], back[line + 1]
        for done in range(size + 1):
            here = row[done]
            if here == infinity:
                continue
            left = size - done
            if left == 0:
                if here + UNPLACED < following[done]:
                    following[done] = here + UNPLACED
                    origin[done] = done
                continue
            for take in range(1, min(LONGEST, left) + 1):
                end = done + take
                if last and end != size:
                    continue
                value = here + abs(logs[take] - want) + rest[end]
                if value < following[end]:
                    following[end] = value
                    origin[end] = done
    end = size if best[lines][size] < infinity else min(range(size + 1), key=lambda index: best[lines][index])
    spans = []
    for line in range(lines, 0, -1):
        begin = back[line][end]
        spans.append((first + begin, first + end))
        end = begin
    spans.reverse()
    return spans


def _shown(parts) -> str:
    """The syllables that share one note as the text drawn under it: a hyphen where the word goes on to a later note."""
    text = ""
    for place, (piece, _word, closes) in enumerate(parts):
        text += piece
        if place + 1 < len(parts) and closes:
            text += " "
    return text if parts[-1][2] else text + "-"


def lay(sheet, lyrics: str) -> dict:
    """The lyrics laid under the vocal notes of a sheet from ``notation.read``.

    ``notes`` holds ``[note, line, text]`` for every note with words under
    it: the note's place among the vocal notes in the order they start, the
    row of ``lines`` it belongs to, and the text. ``spare`` lists the notes
    of a line that no syllable is left for. ``lines`` has a row for every sung
    line: its number in the lyrics as written, its text, the notes it is laid
    on as ``first`` to ``stop``, how many syllables and notes that is, and its
    ``kind`` -- ``FIT``, ``SPARE`` or ``CROWD``. ``counts`` adds the kinds up.

    ``paired`` is False when the sections of the lyrics could not be told from
    the sections of the score and the lines were laid along the whole song.
    ``bare`` is True, with nothing laid, for a score that names no section.
    """
    notes = sorted(sheet["notes"]["Vocal"], key=lambda note: (note["start"], note["pitch"]))
    blocks = blocks_of(lyrics)
    laid = {"bare": False, "paired": True, "lines": [], "notes": [], "spare": [],
            "counts": {FIT: 0, SPARE: 0, CROWD: 0}, "sung": len(notes)}
    if not any(str(section["name"]).strip() for section in sheet["sections"]):
        laid["bare"] = True
        return laid
    if not blocks:
        return laid
    found, laid["paired"] = stretches(sheet, notes, blocks)
    rows = []
    for stretch, first, stop in found:
        spans = cut(notes, [len(row["parts"]) for row in stretch], first, stop, sheet["per_quarter"])
        rows.extend(zip(stretch, spans))
    rows.sort(key=lambda pair: pair[0]["line"])
    for index, (row, (low, high)) in enumerate(rows):
        parts = row["parts"]
        held = high - low
        kind = FIT if held == len(parts) else SPARE if held > len(parts) else CROWD
        laid["counts"][kind] += 1
        laid["lines"].append({"line": row["line"], "text": row["text"], "first": low, "stop": high,
                              "syllables": len(parts), "notes": held, "kind": kind})
        if not held:
            continue
        for place in range(min(held, len(parts))):
            on = parts[place:] if place == held - 1 else parts[place:place + 1]
            laid["notes"].append([low + place, index, _shown(on)])
        laid["spare"].extend(range(low + len(parts), high))
    laid["notes"].sort()
    return laid
