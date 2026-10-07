"""Mending recognised words with a second hearing: the language, the request to a language model, its answer.

With 'second_hearing' on, YuE2 Transcribe hears a song twice: Qwen3-ASR as
always, then Whisper large-v3 turbo told the same language. A language model
gets both texts and writes the lyrics, taking each word from whichever
hearing makes sense, and its answer is checked before it replaces Qwen3's.

Measured on 2026-09-26 and again on 2026-10-01 (M26.0) on fourteen songs,
four real Russian ones with their published lyrics and ten the pack sang,
with Qwen3.6-27B Q5 choosing over three seeds: the real songs' WER went
0.148 -> 0.084-0.102 and the pack's Russian songs 0.112 -> 0.036-0.048,
the English ones 0.068 -> 0.059. Every seed fixed 84-95 true words and
broke 3-5. A 4B model gave about half of that. A model alone, given only
Qwen3's words, the score or time to think, did not help; Whisper alone heard
worse than Qwen3.

The language comes first (``language_plan``). Whisper told the wrong
language translates, and left to itself it takes the language of the first
thirty seconds, so a song that opens without a voice came out as a
translation. Its chance of each language summed over every thirty seconds of
the mix named all fourteen songs right. Qwen3 is heard again, told that
language, when its own label or the letters of its words say otherwise: it
had written one Russian duet as an English translation.

The answer (``judged``) is refused, and Qwen3's words kept, when it is longer
than ``LONGER`` times the longer hearing, shorter than ``SHORTER`` times the
shorter one, or when more than ``NOVEL`` of its words stand where neither
hearing has them. The lower bound caught a 4B answer that dropped a quarter
of a song; at 5 percent new words the 27B's sound answers on one quirky song
were refused too, and at 8 percent none of them were.

Standard library only.
"""

from __future__ import annotations

import difflib
import re

from .. import languages

RULES = (
    "You proofread song lyrics that a speech recogniser wrote down from a sung recording.\n\n"
    "Singing is harder to hear than speech, so the recogniser sometimes writes a word that sounds like the "
    "sung one but makes no sense where it stands, splits or merges words, or gets an ending wrong. Correct "
    "such mishearings, and only them: a corrected word should sound like the word that was written.\n\n"
    "Keep every word that could have been sung as written, even when it is unusual, rude, absurd, slang or "
    "a made-up name: songs are often strange on purpose. Do not rephrase, do not improve style or grammar, "
    "do not translate, do not add or drop lines, keep every repetition. When in doubt, keep the word."
)
TWO = (
    "\n\nYou get what two independent recognisers wrote for the same recording. Where they agree, keep "
    "the words. Where they differ, take the reading that makes sense in the song, from either text, word "
    "by word. Write a word neither of them wrote only when both are clearly wrong in the same place and "
    "your word sounds like theirs.")
ANSWER = "\n\nAnswer with the whole corrected lyrics between <lyrics> and </lyrics>, and nothing after them."

ANSWER_TOKENS = 3000
TEMPERATURE = 0.3
TOP_P = 0.9
TOP_K = 20
REPEAT_PENALTY = 1.0
"""The sampling the measurement ran with, thinking off."""

LONGER = 1.15
SHORTER = 0.9
NOVEL = 0.08

_LYRICS = re.compile(r"<lyrics>(.*?)(?:</lyrics>|$)", re.DOTALL)
_TAG_LINE = re.compile(r"(?m)^[ \t]*\[[^\]\n]*\][^\n]*$")


def spoken(names) -> str:
    """Language names as the request says them: "Russian", "Russian and English"."""
    names = [name for name in names if name]
    if len(names) < 2:
        return names[0] if names else ""
    return ", ".join(names[:-1]) + " and " + names[-1]


def messages(language: str, first: str, second: str) -> list:
    """The chat: the rules, then both hearings, Qwen3's as recogniser A and Whisper's as recogniser B."""
    head = "The words are in {}.\n\n".format(language) if language else ""
    return [{"role": "system", "content": RULES + TWO + ANSWER},
            {"role": "user", "content": head + "Recogniser A:\n" + first + "\n\nRecogniser B:\n" + second}]


def answer_of(text: str) -> str:
    """The lyrics in a model's answer: after any thinking, between the lyrics tags, without braces or tag lines."""
    tail = (text or "").split("</think>", 1)[1] if "</think>" in (text or "") else (text or "")
    found = _LYRICS.search(tail)
    lyrics = (found.group(1) if found else tail).replace("{", "").replace("}", "")
    return _TAG_LINE.sub("", lyrics).strip()


def words(text: str) -> list:
    """Words as they are compared: lower case, yo as ye, bracketed tags left out."""
    lowered = (text or "").lower().replace("\u0451", "\u0435")
    return re.findall(r"[\w']+", re.sub(r"\[[^\]]*\]", " ", lowered))


def _kept(source: list, target: list) -> list:
    """For each word of ``target``, whether the alignment finds it unchanged in ``source``."""
    kept = [False] * len(target)
    for tag, _i1, _i2, first, last in difflib.SequenceMatcher(a=source, b=target, autojunk=False).get_opcodes():
        if tag == "equal":
            for index in range(first, last):
                kept[index] = True
    return kept


def judged(first: str, second: str, answer: str) -> dict:
    """``{"text", "refused", "taken"}``: the answer, or ``first`` with the reason it was refused.

    ``taken`` counts the answer's words that stand where Whisper has them and
    Qwen3 does not: the words the second hearing mended.
    """
    a, b, c = words(first), words(second), words(answer)
    if not c:
        return {"text": first, "refused": "empty", "taken": 0}
    from_first, from_second = _kept(a, c), _kept(b, c)
    novel = sum(1 for x, y in zip(from_first, from_second) if not (x or y)) / len(c)
    if len(c) / max(len(a), len(b), 1) > LONGER:
        refused = "longer"
    elif len(c) / max(min(len(a), len(b)), 1) < SHORTER:
        refused = "shorter"
    elif novel > NOVEL:
        refused = "novel"
    else:
        taken = sum(1 for x, y in zip(from_first, from_second) if y and not x)
        return {"text": answer, "refused": "", "taken": taken, "novel": novel}
    return {"text": first, "refused": refused, "taken": 0, "novel": novel}


def candidates() -> list:
    """The Whisper codes of the languages Qwen3-ASR knows: a song's language is chosen among those."""
    from .prompt import LANGUAGES

    return [languages.whisper_code(name) for name in LANGUAGES.values() if languages.whisper_code(name)]


def language_plan(heard: dict, windows: list) -> tuple:
    """``(language, again)`` for a song left at 'auto': the language to tell both hearings, and whether Qwen3 hears it again.

    ``heard`` is Qwen3's ``{"language", "text"}``, ``windows`` Whisper's
    language windows (``whisper.model.identify``). The language is Whisper's
    summed over every window, among Qwen3's languages; with no window to go
    by it is Qwen3's own label. Qwen3 hears the song again, told it, when its
    label names another language or its words are written in another script.
    Both are language names; "" when neither model names one Qwen3 knows.
    """
    from ..whisper import decode

    label = languages.qwen_name(languages.known(heard.get("language") or "")) if heard.get("language") else ""
    code = decode.best_language(windows, candidates()) if windows else ""
    language = languages.known(code) if code else label
    if not language:
        return "", False
    again = label != language or languages.script_of(heard.get("text", "")) != languages.script_of_language(language)
    return language, again
