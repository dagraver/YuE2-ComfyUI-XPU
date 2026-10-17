"""Whisper kept between runs: what it heard is kept, a small card takes turns, Unload Models lets it go."""

from __future__ import annotations

import sys
import types

import pytest

from yue2_comfy import memory
from yue2_comfy.asr import runtime as speech
from yue2_comfy.whisper import runtime


@pytest.fixture
def whisper(monkeypatch):
    """Whisper's model stood in: what was loaded, heard and named, and whether the captured step was let go."""
    from yue2_comfy.whisper import tokenizer

    events = []
    net = types.SimpleNamespace(forget_steps=lambda: events.append("forgot"))
    fake = types.SimpleNamespace(
        load=lambda folder, device: events.append("loaded") or net,
        vocabulary=lambda folder: "vocab",
        mono_16k=lambda waveform, rate: waveform,
        hear=lambda net, vocab, tok, audio, code, cancelled=None, progress=None:
            events.append(("heard", audio, code)) or {"text": audio + " in " + code},
        identify=lambda net, vocab, audio, cancelled=None, progress=None:
            events.append(("named", audio)) or [{"ru": -0.1}])
    monkeypatch.setitem(sys.modules, "yue2_comfy.whisper.model", fake)
    monkeypatch.setattr(sys.modules["yue2_comfy.whisper"], "model", fake, raising=False)
    monkeypatch.setattr(tokenizer, "Tokenizer", lambda path: "tokenizer")
    monkeypatch.setattr(runtime, "stamp", lambda folder: ("weights", folder))
    monkeypatch.setattr(runtime, "_STATE", {"key": None, "net": None, "tokenizer": None, "vocab": None})
    monkeypatch.setattr(runtime, "_HEARD", runtime.collections.OrderedDict())
    monkeypatch.setattr(runtime, "_IDENTIFIED", runtime.collections.OrderedDict())
    return events


def test_what_whisper_heard_is_kept_by_the_recording_and_the_language(whisper):
    assert runtime.hear("f", "cpu", "song", 16000, "mark", "ru") == {"text": "song in ru"}
    runtime.hear("f", "cpu", "song", 16000, "mark", "ru")
    runtime.hear("f", "cpu", "song", 16000, "mark", "en")
    runtime.hear("f", "cpu", "other", 16000, "other mark", "ru")
    assert [event for event in whisper if isinstance(event, tuple)] == [
        ("heard", "song", "ru"), ("heard", "song", "en"), ("heard", "other", "ru")]
    assert whisper.count("loaded") == 1


def test_new_weights_are_a_new_hearing(whisper):
    runtime.hear("f", "cpu", "song", 16000, "mark", "ru")
    runtime.hear("g", "cpu", "song", 16000, "mark", "ru")
    assert [event for event in whisper if isinstance(event, tuple)] == [("heard", "song", "ru")] * 2


def test_the_language_windows_are_kept_by_the_recording(whisper):
    assert runtime.identify("f", "cpu", "song", 16000, "mark") == [{"ru": -0.1}]
    runtime.identify("f", "cpu", "song", 16000, "mark")
    assert [event for event in whisper if isinstance(event, tuple)] == [("named", "song")]


def test_the_captured_step_is_let_go_however_the_hearing_ends(whisper, monkeypatch):
    runtime.hear("f", "cpu", "song", 16000, "mark", "ru")
    assert whisper[-1] == "forgot"

    def broken(*args, **kwargs):
        raise InterruptedError("cancelled")

    monkeypatch.setattr(sys.modules["yue2_comfy.whisper.model"], "hear", broken)
    with pytest.raises(InterruptedError):
        runtime.hear("f", "cpu", "song", 16000, "mark 2", "ru")
    assert whisper[-1] == "forgot"
    assert runtime.is_loaded()
    runtime.unload()
    assert not runtime.is_loaded()


def test_a_small_card_lets_the_speech_model_and_its_aligner_go_first(whisper, monkeypatch):
    gone = []
    monkeypatch.setattr(speech, "unload", lambda: gone.append("Qwen3-ASR"))
    monkeypatch.setattr(speech, "unload_aligner", lambda: gone.append("aligner"))
    runtime.acquire("f", "cpu")
    assert gone == []
    runtime.unload()
    runtime.acquire("f", "cpu", low_vram=True)
    assert gone == ["Qwen3-ASR", "aligner"] and runtime.is_loaded()


def test_a_small_card_lets_whisper_go_before_the_speech_model_or_the_aligner_comes(monkeypatch):
    from yue2_comfy.asr import tokenizer

    fake_model = types.SimpleNamespace(load=lambda folder, device, low_vram=False: types.SimpleNamespace())
    fake_aligner = types.SimpleNamespace(load=lambda folder, reader, device, low_vram=False: "held")
    for name, fake in (("model", fake_model), ("aligner", fake_aligner)):
        monkeypatch.setitem(sys.modules, "yue2_comfy.asr." + name, fake)
        monkeypatch.setattr(sys.modules["yue2_comfy.asr"], name, fake, raising=False)
    monkeypatch.setattr(tokenizer, "Tokenizer", lambda path: "tokenizer")
    monkeypatch.setattr(speech, "stamp", lambda folder: ("weights",))
    monkeypatch.setattr(speech, "_STATE", {"key": None, "net": None, "tokenizer": None})
    monkeypatch.setattr(speech, "_ALIGNER", {"key": None, "held": None})
    gone = []
    monkeypatch.setattr(runtime, "unload", lambda: gone.append("Whisper"))
    speech.acquire("speech", "cpu")
    speech.acquire_aligner("aligner", "reader", "cpu")
    assert gone == []
    speech.acquire("speech", "cpu", low_vram=True)
    speech.acquire_aligner("aligner", "reader", "cpu", low_vram=True)
    assert gone == ["Whisper", "Whisper"]


def test_a_short_card_lets_whisper_go_after_the_other_listeners_and_before_sheetsage(monkeypatch):
    from yue2_comfy import loader
    from yue2_comfy.sheetsage import runtime as sheetsage_runtime

    management = types.SimpleNamespace(unload_all_models=lambda: None, soft_empty_cache=lambda force=False: None)
    monkeypatch.setitem(sys.modules, "comfy", types.SimpleNamespace(model_management=management))
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    monkeypatch.setattr(speech, "_free_bytes", lambda device: 0)
    gone = []
    for name, module in (("Whisper", runtime), ("SheetSage2", sheetsage_runtime), ("YuE2", loader)):
        monkeypatch.setattr(module, "is_loaded", lambda: True)
        monkeypatch.setattr(module, "unload", lambda name=name: gone.append(name))
    speech._make_room(types.SimpleNamespace(type="cuda"), speech.ROOM_BYTES,
                      [("aligner", lambda: True, lambda: gone.append("aligner")),
                       ("Qwen3-ASR", lambda: True, lambda: gone.append("Qwen3-ASR"))])
    assert gone[:4] == ["aligner", "Qwen3-ASR", "Whisper", "SheetSage2"]


def test_unload_models_lets_whisper_go():
    assert ("Whisper", ".whisper.runtime") in memory.keepers()
    assert callable(runtime.unload) and callable(runtime.is_loaded)
