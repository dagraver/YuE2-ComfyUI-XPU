"""The 'language' field: what a person may write, and what the speech models are told."""

from __future__ import annotations

import pytest

from yue2_comfy import languages
from yue2_comfy.asr import prompt


@pytest.mark.parametrize("text", ["", "   ", "auto", "AUTO", " Auto ", None])
def test_auto_or_nothing_names_no_language(text):
    assert languages.parse(text) == ()


@pytest.mark.parametrize("text, names", [
    ("Russian", ("Russian",)),
    ("ru", ("Russian",)),
    ("  rUsSiAn ", ("Russian",)),
    ("Russian, English", ("Russian", "English")),
    ("ru,en", ("Russian", "English")),
    ("ru; en", ("Russian", "English")),
    ("English, Russian, english", ("English", "Russian")),
    ("Russian,, English,", ("Russian", "English")),
    ("haitian   creole", ("Haitian Creole",)),
    ("Ukrainian", ("Ukrainian",)),
    ("uk, ru", ("Ukrainian", "Russian")),
])
def test_names_and_codes_are_read_in_order_once_each(text, names):
    assert languages.parse(text) == names


def test_tagalog_and_filipino_are_one_language():
    assert languages.parse("Tagalog") == languages.parse("tl") == languages.parse("fil") == ("Filipino",)
    assert languages.whisper_code("Filipino") == "tl"
    assert languages.qwen_name("Filipino") == "Filipino"


def test_a_name_no_language_has_is_refused_with_how_to_write_one():
    with pytest.raises(ValueError) as error:
        languages.parse("Russian, Klingon")
    message = str(error.value)
    assert "'Klingon'" in message and "'Russian, English'" in message and "'auto'" in message


def test_auto_among_names_is_refused():
    with pytest.raises(ValueError, match="'auto' alone"):
        languages.parse("auto, Russian")


def test_every_language_the_speech_model_knows_has_a_whisper_code():
    for code, name in prompt.LANGUAGES.items():
        assert languages.known(code) == name and languages.known(name) == name
        assert languages.whisper_code(name), name
        assert languages.qwen_name(name) == name


def test_a_language_only_whisper_knows_is_not_told_to_the_speech_model():
    assert languages.qwen_name("Ukrainian") == ""
    assert languages.whisper_code("Ukrainian") == "uk"
    assert languages.script_of_language("Ukrainian") == ""


def test_whispers_table_is_one_code_and_one_name_a_language():
    codes = [code for code, _name in languages.WHISPER_LANGUAGES]
    names = [name for _code, name in languages.WHISPER_LANGUAGES]
    assert len(codes) == len(set(codes)) == 100
    assert len(names) == len(set(names))


@pytest.mark.parametrize("text, script", [
    ("\u0413\u043e\u0440\u043e\u0434 \u0441\u043f\u0438\u0442", "Cyrillic"),
    ("The city sleeps", "Latin"),
    ("\u0413\u043e\u0440\u043e\u0434 and a city \u0441\u043f\u0438\u0442 tonight", "Latin"),
    ("\u4f60\u597d\u4e16\u754c", "Han"),
    ("\u3053\u3093\u306b\u3061\u306f\u4e16\u754c", "Kana"),
    ("\uc548\ub155\ud558\uc138\uc694", "Hangul"),
    ("123 ... !", ""),
])
def test_the_script_of_a_text_is_the_one_most_of_its_letters_are_in(text, script):
    assert languages.script_of(text) == script


def test_each_speech_model_language_has_the_script_its_words_come_in():
    assert languages.script_of_language("Russian") == "Cyrillic"
    assert languages.script_of_language("English") == "Latin"
    assert languages.script_of_language("Polish") == "Latin"
    assert languages.script_of_language("Japanese") == "Kana"
    assert languages.script_of_language("Cantonese") == "Han"
