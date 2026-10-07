"""The languages a song is sung in: what the 'language' field on YuE2 Transcribe may name. Standard library only.

A song's words are heard by Qwen3-ASR, which knows thirty languages and is
told one by name (``asr.prompt.LANGUAGES``). Left to tell the language itself,
it can name the wrong one and then write the whole song as a translation into
it -- measured on 2026-10-01: a Russian duet came out in English, word for
word translated -- so the person who knows the song can name it instead.

A name is written in English or as a code, case and spacing aside, and
several are separated by commas. The field also takes Whisper's languages,
which Qwen3-ASR does not know (Ukrainian, Hebrew, ...): the speech model then
tells the language itself, and a second hearing, where there is one, is told
the named one. Filipino and Tagalog are one language under two names.

'auto', or nothing, leaves the language to the speech model, which is how the
node heard every song before the field existed.
"""

from __future__ import annotations

import unicodedata

from .asr.prompt import LANGUAGES as QWEN_LANGUAGES

AUTO = "auto"

WHISPER_LANGUAGES = (
    ("en", "English"), ("zh", "Chinese"), ("de", "German"), ("es", "Spanish"), ("ru", "Russian"),
    ("ko", "Korean"), ("fr", "French"), ("ja", "Japanese"), ("pt", "Portuguese"), ("tr", "Turkish"),
    ("pl", "Polish"), ("ca", "Catalan"), ("nl", "Dutch"), ("ar", "Arabic"), ("sv", "Swedish"),
    ("it", "Italian"), ("id", "Indonesian"), ("hi", "Hindi"), ("fi", "Finnish"), ("vi", "Vietnamese"),
    ("he", "Hebrew"), ("uk", "Ukrainian"), ("el", "Greek"), ("ms", "Malay"), ("cs", "Czech"),
    ("ro", "Romanian"), ("da", "Danish"), ("hu", "Hungarian"), ("ta", "Tamil"), ("no", "Norwegian"),
    ("th", "Thai"), ("ur", "Urdu"), ("hr", "Croatian"), ("bg", "Bulgarian"), ("lt", "Lithuanian"),
    ("la", "Latin"), ("mi", "Maori"), ("ml", "Malayalam"), ("cy", "Welsh"), ("sk", "Slovak"),
    ("te", "Telugu"), ("fa", "Persian"), ("lv", "Latvian"), ("bn", "Bengali"), ("sr", "Serbian"),
    ("az", "Azerbaijani"), ("sl", "Slovenian"), ("kn", "Kannada"), ("et", "Estonian"), ("mk", "Macedonian"),
    ("br", "Breton"), ("eu", "Basque"), ("is", "Icelandic"), ("hy", "Armenian"), ("ne", "Nepali"),
    ("mn", "Mongolian"), ("bs", "Bosnian"), ("kk", "Kazakh"), ("sq", "Albanian"), ("sw", "Swahili"),
    ("gl", "Galician"), ("mr", "Marathi"), ("pa", "Punjabi"), ("si", "Sinhala"), ("km", "Khmer"),
    ("sn", "Shona"), ("yo", "Yoruba"), ("so", "Somali"), ("af", "Afrikaans"), ("oc", "Occitan"),
    ("ka", "Georgian"), ("be", "Belarusian"), ("tg", "Tajik"), ("sd", "Sindhi"), ("gu", "Gujarati"),
    ("am", "Amharic"), ("yi", "Yiddish"), ("lo", "Lao"), ("uz", "Uzbek"), ("fo", "Faroese"),
    ("ht", "Haitian Creole"), ("ps", "Pashto"), ("tk", "Turkmen"), ("nn", "Nynorsk"), ("mt", "Maltese"),
    ("sa", "Sanskrit"), ("lb", "Luxembourgish"), ("my", "Myanmar"), ("bo", "Tibetan"), ("tl", "Filipino"),
    ("mg", "Malagasy"), ("as", "Assamese"), ("tt", "Tatar"), ("haw", "Hawaiian"), ("ln", "Lingala"),
    ("ha", "Hausa"), ("ba", "Bashkir"), ("jw", "Javanese"), ("su", "Sundanese"), ("yue", "Cantonese"),
)
"""Whisper's languages by the code its tokenizer writes, named as Qwen3-ASR names those it shares.

Whisper calls ``tl`` Tagalog, the language Qwen3-ASR calls Filipino; it is
kept under Qwen's name and Tagalog is an alias.
"""

ALIASES = {"tagalog": "Filipino", "fil": "Filipino", "burmese": "Myanmar"}

SCRIPTS = {
    "Russian": "Cyrillic", "Macedonian": "Cyrillic", "Arabic": "Arabic", "Persian": "Arabic", "Greek": "Greek",
    "Hindi": "Devanagari", "Thai": "Thai", "Korean": "Hangul", "Japanese": "Kana", "Chinese": "Han",
    "Cantonese": "Han",
}
"""The script each of Qwen3-ASR's languages is written in; the rest of its thirty are written in Latin letters."""

UNKNOWN = (
    "'language' does not know '{name}'. Write the languages the song is sung in, in English or as codes and "
    "separated by commas -- 'Russian', 'Russian, English', 'ru, en' -- or 'auto' to let the speech model tell."
)
AUTO_AMONG = "'language' is either 'auto' alone or a list of languages; it cannot be both."


def _key(text: str) -> str:
    return " ".join(str(text).lower().split())


def _table() -> dict:
    names = {}
    for code, name in WHISPER_LANGUAGES:
        names[_key(code)] = name
        names[_key(name)] = name
    for code, name in QWEN_LANGUAGES.items():
        names[_key(code)] = name
        names[_key(name)] = name
    for alias, name in ALIASES.items():
        names[_key(alias)] = name
    return names


NAMES = _table()


def known(text: str) -> str:
    """The language a name or code means, as this module names it, or "" for none."""
    return NAMES.get(_key(text), "")


def parse(text) -> tuple:
    """The languages the field names, in the order written, each once; () for 'auto' or nothing.

    ValueError, with the words a person needs to fix it, for a name no
    language has or for 'auto' written among names.
    """
    raw = str(text or "").strip()
    if _key(raw) in ("", AUTO):
        return ()
    found = []
    for piece in raw.replace(";", ",").split(","):
        if not piece.strip():
            continue
        if _key(piece) == AUTO:
            raise ValueError(AUTO_AMONG)
        name = known(piece)
        if not name:
            raise ValueError(UNKNOWN.format(name=piece.strip()))
        if name not in found:
            found.append(name)
    return tuple(found)


def qwen_name(name: str) -> str:
    """The name Qwen3-ASR is told for this language, or "" when it does not know it."""
    return name if name in QWEN_LANGUAGES.values() else ""


def whisper_code(name: str) -> str:
    """The code Whisper's tokenizer writes for this language, or "" when it has none."""
    for code, known_name in WHISPER_LANGUAGES:
        if known_name == name:
            return code
    return ""


def script_of_language(name: str) -> str:
    """The script one of Qwen3-ASR's languages is written in, or "" for a language it does not know."""
    if not qwen_name(name):
        return ""
    return SCRIPTS.get(name, "Latin")


def script_of(text: str) -> str:
    """The script most of the text's letters are written in, as ``SCRIPTS`` names them; "" for no letters."""
    counts = {}
    for character in text or "":
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        if name.startswith(("HIRAGANA", "KATAKANA")):
            kind = "Kana"
        elif name.startswith("CJK"):
            kind = "Han"
        else:
            kind = name.split(" ", 1)[0].title()
        counts[kind] = counts.get(kind, 0) + 1
    if counts.get("Kana"):
        counts["Kana"] += counts.pop("Han", 0)
    return max(counts, key=counts.get) if counts else ""
