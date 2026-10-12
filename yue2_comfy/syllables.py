"""A word of lyrics cut into the syllables it is sung on, each a piece of the word's own letters.

YuE2 writes about one note for every syllable of a line (see ``underlay``), so
words drawn under notes need the syllables themselves, and a count that is
right more often than a count to lay lines out by has to be.

``phrasing`` counts syllables too, to lay lines along a tune from a MIDI file,
and its count stays as it is: a different count there would move the bars the
songs of 'YuE2 Load MIDI' are sung on. Measured on 2026-10-12 on 280 words
songs are made of, counted by hand, that count was off on 48 -- "didn't",
"lonely", "something", "neon", "candles" -- and on 36 of 168 more that no rule
had been written against. The rules here are that count and what those words
asked for:

- a word that ends in ``n't`` after a consonant sings the ``n't`` as a
  syllable of its own, "did-n't"; the other short endings -- ``'s``, ``'ll``,
  ``'ve``, ``'re``, ``'d``, ``'m`` -- are taken off before the word is cut and
  go with its last syllable, or after a consonant are one themselves,
  "could-'ve";
- an e before a common ending is as silent as at the end of a word:
  "lone-ly", "some-thing", "care-less", "life-time";
- ``-les`` after a consonant is sung, "can-dles", and so is ``-ue`` unless a g
  or a q stands before it, "val-ue" but "tongue";
- words whose letters do not say how they are sung are spelled out in
  ``SPELLED``: "ne-on", "qui-et", "i-de-a", "eve-ry".

Cyrillic is cut at its vowels, a vowel a syllable, which is exact; a word of
one or two letters with no vowel in it -- a preposition, a particle -- is no
syllable and is sung with its neighbour. Chinese, Japanese and Korean are a
syllable a character. Where a cut falls between two vowels is a matter of
reading, not of singing: one consonant goes with the vowel after it, and of
several the last does, or the last two when they begin words together.

Standard library only.
"""

from __future__ import annotations

import unicodedata

from . import phrasing

WORD = phrasing.WORD
"""What a word is: letters, with an apostrophe inside it kept. The rule ``phrasing`` counts by."""

LATIN_VOWELS = "aeiouy\u00e6\u00f8\u0153"
GREEK_VOWELS = "\u03b1\u03b5\u03b7\u03b9\u03bf\u03c5\u03c9"
SEPARATE = "\u00ef\u00eb\u00fc\u00ff"
"""A vowel written with two dots after another vowel starts a syllable of its own."""

SILENT = "\u044c\u044a'"
"""Letters of a Cyrillic word that are no sound of their own: the two signs and an apostrophe."""

TAILS = ("n't", "'ll", "'ve", "'re", "'s", "'d", "'m")
"""Short endings taken off a word before it is cut."""

SUNG_TAILS = ("'ll", "'ve", "'re", "'d")
"""The ones sung as a syllable after a consonant: "that'll", "could've", "it'd"."""

SUFFIXES = ("fully", "ly", "ful", "ments", "ment", "less", "ness", "times", "time", "things", "thing", "where",
            "body", "lands", "land", "work", "days", "day", "side", "self", "like", "some", "how", "what")
"""Endings an e before them is silent in front of, when it follows one consonant: "lone-ly", "some-where"."""

ONSETS = frozenset({"th", "sh", "ch", "ph", "wh", "bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl",
                    "pr", "tr"})
"""Pairs of consonants that begin a syllable together when they stand between two vowels."""

HUSHED = ("s", "x", "z", "ce", "se", "ge", "ze", "ch", "sh")
"""Endings an added s is sung after as a syllable, so a spelled word's plural is not read off its singular."""

SPELLED = {word.replace("-", ""): word for word in """
may-be be-yond ne-on vid-e-o qui-et qui-et-ly dia-mond po-em po-et po-et-ry i-de-a cre-ate cre-a-ted
cre-a-tion crea-ture re-al-i-ty so-ci-e-ty va-ri-e-ty an-xi-e-ty sci-ence rhy-thm clothes a-ny-one
some-one eve-ry-one real real-ly a-re-a o-a-sis cha-os tri-al gi-ant cli-ent fu-el cru-el jew-el du-et
ru-in flu-id ge-nius au-di-o stu-di-o ro-de-o ste-re-o mu-se-um the-a-ter the-a-tre hal-le-lu-jah
na-ive he-ro he-roes e-cho e-choes prayer fire hour our light-ning ar-gue eve-ry eve-ry-thing
eve-ry-bo-dy eve-ry-where eve-ry-day eve-ning eve-nings diffe-rent diffe-rence diffe-rent-ly seve-ral
favo-rite favou-rite came-ra choco-late inte-rest inte-rest-ing busi-ness vege-ta-ble comfor-ta-ble
tempe-ra-ture lone-li-ness despe-rate despe-rate-ly
""".split()}
"""Words cut by hand, as they are sung: the ones whose letters mislead the rules, and the ones singers shorten.

"Loneliness" and "desperately" were the two words still counted wrong when
the rules were measured, one of the 280 the rules were written against and
one of the 168 they were not; they were spelled out afterwards.
"""


def _wide(character: str) -> bool:
    return ("\u4e00" <= character <= "\u9fff" or "\u3400" <= character <= "\u4dbf"
            or "\u3040" <= character <= "\u30ff" or "\uac00" <= character <= "\ud7af")


def _base(character: str) -> str:
    """A letter without its accent, so a word reads the same length with or without them."""
    for part in unicodedata.normalize("NFD", character):
        if not unicodedata.combining(part):
            return part
    return character


def _runs(letters: str, vowels: str) -> list:
    """``[start, stop]`` for every run of vowels in ``letters``."""
    found = []
    opened = None
    for index, letter in enumerate(letters):
        if letter in vowels:
            if opened is None:
                opened = index
        elif opened is not None:
            found.append([opened, index])
            opened = None
    if opened is not None:
        found.append([opened, len(letters)])
    return found


def _part(groups: list, index: int) -> None:
    """Cut the run that holds letter ``index`` in two there, unless it already starts at it."""
    for place, (start, stop) in enumerate(groups):
        if start < index < stop:
            groups[place:place + 1] = [[start, index], [index, stop]]
            return


def _sliced(text: str, cuts: list) -> list:
    edges = [0] + sorted(set(cut for cut in cuts if 0 < cut < len(text))) + [len(text)]
    return [text[low:high] for low, high in zip(edges, edges[1:])]


def _cyrillic(text: str, lowered: str) -> list:
    nuclei = _runs(lowered, phrasing.CYRILLIC_VOWELS)
    parted = [[index, index + 1] for start, stop in nuclei for index in range(start, stop)]
    if not parted:
        return [] if len([letter for letter in lowered if letter not in SILENT]) <= 2 else [text]
    cuts = []
    for left, right in zip(parted, parted[1:]):
        low, high = left[1], right[0]
        sounded = [letter for letter in lowered[low:high] if letter not in SILENT]
        if len(sounded) < 2:
            cuts.append(low)
            continue
        cut = low + 1
        while cut < high and lowered[cut] in SILENT:
            cut += 1
        cuts.append(cut)
    return _sliced(text, cuts)


def _wide_pieces(text: str) -> list:
    found = []
    held = ""
    for character in text:
        if _wide(character):
            found.append(held + character)
            held = ""
        elif found:
            found[-1] += character
        else:
            held += character
    return found or [text]


def _spelled(plain: str):
    """The hand-made cut of a word, or of the word it is the plural of; None for any other."""
    if plain in SPELLED:
        return SPELLED[plain]
    if plain.endswith("s") and plain[:-1] in SPELLED and not plain[:-1].endswith(HUSHED):
        return SPELLED[plain[:-1]] + "s"
    return None


def _latin(plain: str, lowered: str) -> list:
    """Where a word in Latin letters is cut, as letter indexes; ``plain`` is it without accents."""
    size = len(plain)
    spelled = _spelled(plain)
    if spelled is not None:
        cuts, at = [], 0
        for piece in spelled.split("-")[:-1]:
            at += len(piece)
            cuts.append(at)
        return cuts
    groups = _runs(plain, LATIN_VOWELS)
    for index in range(1, size):
        if lowered[index] in SEPARATE and plain[index - 1] in LATIN_VOWELS:
            _part(groups, index)
    if plain.endswith("ing") and size > 3 and plain[-4] in "aeiouy":
        _part(groups, size - 3)
    for found in phrasing.SOUNDED_I.finditer(plain):
        _part(groups, found.end())

    silent = set()
    for suffix in SUFFIXES:
        stem = plain[:size - len(suffix)]
        if (plain.endswith(suffix) and len(stem) >= 3 and stem.endswith("e") and stem[-2] not in "aeiouy"
                and any(letter in "aeiouy" for letter in stem[:-2]) and not stem.endswith("le")
                and [len(stem) - 1, len(stem)] in groups):
            silent.add(len(stem) - 1)
            break
    sounded = [group for group in groups if group[0] not in silent]
    syllabic = None
    if len(sounded) > 1 and size > 2:
        last = sounded[-1]
        if plain.endswith("e") and not lowered.endswith(phrasing.STRESSED_E):
            if plain.endswith("le") and plain[-3] not in "aeiouy":
                syllabic = last
            elif last == [size - 1, size] or plain.endswith(("gue", "que")):
                silent.update(range(last[0], last[1]))
        elif plain.endswith("es") and size > 3 and plain[-3] not in "aeiouysxzcgh":
            if plain.endswith("les") and size > 4 and plain[-4] not in "aeiouy":
                syllabic = last
            else:
                silent.add(size - 2)
        elif plain.endswith("ed") and size > 3 and plain[-3] not in "aeiouytd":
            silent.add(size - 2)
    sounded = [group for group in groups if group[0] not in silent]

    cuts = []
    for left, right in zip(sounded, sounded[1:]):
        low, high = left[1], right[0]
        quiet = [index for index in range(low, high) if index in silent]
        if high == low:
            cuts.append(high)
        elif quiet:
            cuts.append(quiet[-1] + 1)
        elif high - low == 1:
            cuts.append(low)
        elif plain[high - 2:high] == "gh" or (plain[high - 2:high] == "ng" and plain.startswith("ing", high)):
            cuts.append(high)
        elif plain[high - 2:high] in ONSETS or (right is syllabic and plain[high - 1] == "l") \
                or (high - low > 2 and plain[high - 2] == "s" and plain[high - 1] in "ptckw"):
            cuts.append(high - 2)
        else:
            cuts.append(high - 1)
    return cuts


def pieces(word: str) -> list:
    """``word`` cut into the syllables it is sung on, as it is written; joined, they are the word again.

    An empty list for a Cyrillic word of one or two letters with no vowel,
    which is sung with the word beside it. A word no rule here reads -- no
    vowel in it, or letters of a script not listed above -- is one piece.
    """
    text = str(word or "")
    if not text:
        return []
    lowered = text.lower().replace("\u2019", "'")
    if len(lowered) != len(text):
        lowered = "".join(character.lower()[:1] or character for character in text).replace("\u2019", "'")
    if any("\u0400" <= character <= "\u04ff" for character in lowered):
        return _cyrillic(text, lowered)
    if any(_wide(character) for character in text):
        return _wide_pieces(text)

    stem = len(lowered)
    sung = []
    for _turn in range(2):
        for tail in TAILS:
            if lowered[:stem].endswith(tail) and stem > len(tail):
                stem -= len(tail)
                sung.insert(0, tail)
                break
    plain = "".join(_base(character) for character in lowered[:stem])
    if len(plain) != stem:
        plain = lowered[:stem]
    greek = any("\u0370" <= character <= "\u03ff" for character in plain)
    if greek:
        groups = _runs(plain, GREEK_VOWELS)
        cuts = [left[1] if right[0] - left[1] < 2 else right[0] - 1 for left, right in zip(groups, groups[1:])]
    else:
        cuts = _latin(plain, lowered[:stem])
        groups = _runs(plain, LATIN_VOWELS)
    at = stem
    for tail in sung:
        before = plain[-1:] if at == stem else ""
        closed = bool(before) and before not in "aeiouyw"
        if tail == "n't":
            closed = (closed or plain.endswith("e")) and not plain.endswith("re")
        elif tail not in SUNG_TAILS:
            closed = False
        if closed and groups:
            cuts.append(at)
        at += len(tail)
    return _sliced(text, cuts)


def count(word: str) -> int:
    """How many syllables ``word`` is sung on."""
    return len(pieces(word))


def words(line: str) -> list:
    """The words of a line of lyrics, as they are written."""
    return WORD.findall(str(line or ""))


def of_line(line: str) -> int:
    """How many syllables a line is sung on: 0 for a line with no word in it, at least 1 for any other."""
    found = words(line)
    return max(1, sum(count(word) for word in found)) if found else 0
