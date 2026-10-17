"""Mending recognised words with a second hearing, without a model: the request, the answer, the check, the language."""

from __future__ import annotations

import pytest

from yue2_comfy.asr import mend

RUSSIAN = "\u0413\u043e\u0440\u043e\u0434 \u0441\u043f\u0438\u0442 \u043f\u043e\u0434 \u0441\u043d\u0435\u0433\u043e\u043c"


def test_the_request_is_the_measured_one_word_for_word():
    chat = mend.messages("Russian", "a b c", "a b d")
    assert chat[0] == {"role": "system", "content": mend.RULES + mend.TWO + mend.ANSWER}
    assert chat[1]["content"] == "The words are in Russian.\n\nRecogniser A:\na b c\n\nRecogniser B:\na b d"
    assert mend.RULES.startswith("You proofread song lyrics") and "between <lyrics> and </lyrics>" in mend.ANSWER


def test_languages_are_said_as_a_list():
    assert mend.spoken(["Russian"]) == "Russian"
    assert mend.spoken(["Russian", "English"]) == "Russian and English"
    assert mend.spoken(["Russian", "English", "Spanish"]) == "Russian, English and Spanish"
    assert mend.spoken([]) == ""


@pytest.mark.parametrize("raw, lyrics", [
    ("<lyrics>\nla la\nlo lo\n</lyrics>", "la la\nlo lo"),
    ("<think>maybe {x}</think>\n<lyrics>la {la}</lyrics> thanks", "la la"),
    ("<lyrics>[Verse 1]\nla la\n[Chorus]\nlo</lyrics>", "la la\n\nlo"),
    ("<lyrics>la la, cut off", "la la, cut off"),
    ("just the words", "just the words"),
    ("", ""),
])
def test_the_lyrics_are_read_from_the_answer(raw, lyrics):
    assert mend.answer_of(raw) == lyrics


def text(count, word="w"):
    return " ".join("{}{}".format(word, index) for index in range(count))


def test_an_answer_from_both_hearings_stands_and_counts_whispers_words():
    first = "a b c d e f g h i j"
    second = "a b x d e f g h y j"
    verdict = mend.judged(first, second, "a b x d e f g h i j")
    assert verdict["refused"] == "" and verdict["text"] == "a b x d e f g h i j" and verdict["taken"] == 1


def test_an_answer_longer_than_both_hearings_is_refused():
    first, second = text(100), text(100)
    assert mend.judged(first, second, text(115))["refused"] != "longer"
    assert mend.judged(first, second, text(116))["refused"] == "longer"
    assert mend.judged(first, second, text(116))["text"] == first


def test_an_answer_that_drops_words_both_hearings_have_is_refused():
    """A 4B answer that left out a quarter of a song was 0.73 of it; the next shortest answer was 0.98."""
    first, second = text(100), text(110)
    assert mend.judged(first, second, text(90))["refused"] == ""
    assert mend.judged(first, second, text(73))["refused"] == "shorter"


def test_an_answer_with_words_neither_hearing_has_is_refused_past_eight_in_a_hundred():
    first = second = text(100)
    eight = text(92) + " " + text(8, "new")
    nine = text(91) + " " + text(9, "new")
    assert mend.judged(first, second, eight)["refused"] == ""
    assert mend.judged(first, second, nine)["refused"] == "novel"


def test_an_empty_answer_is_refused():
    assert mend.judged("a b", "a c", "")["refused"] == "empty"


def test_words_are_compared_without_case_yo_or_tags():
    assert mend.words("[Verse] \u0401\u043b\u043a\u0430, Don't stop!") == ["\u0435\u043b\u043a\u0430", "don't", "stop"]


def test_the_language_is_chosen_among_the_languages_qwen_knows():
    codes = mend.candidates()
    assert len(codes) == 30 and "tl" in codes and "yue" in codes and "uk" not in codes


def windows(**shares):
    return [dict({"en": -5.0, "ru": -5.0, "uk": -5.0}, **shares)]


@pytest.mark.parametrize("heard, found, plan", [
    ({"language": "Russian", "text": RUSSIAN}, windows(ru=-0.1), ("Russian", False)),
    ({"language": "English", "text": RUSSIAN}, windows(ru=-0.1), ("Russian", True)),
    ({"language": "English", "text": "The city sleeps"}, windows(ru=-0.1), ("Russian", True)),
    ({"language": "Russian", "text": "The city sleeps"}, windows(ru=-0.1), ("Russian", True)),
    ({"language": "English", "text": "The city sleeps"}, windows(en=-0.1), ("English", False)),
    ({"language": "Russian", "text": RUSSIAN}, windows(uk=-0.1, ru=-1.0), ("Russian", False)),
    ({"language": "Russian", "text": RUSSIAN}, [], ("Russian", False)),
    ({"language": "", "text": RUSSIAN}, [], ("", False)),
])
def test_whispers_language_over_the_song_decides_and_qwen_hears_again_when_its_words_disagree(heard, found, plan):
    assert mend.language_plan(heard, found) == plan
