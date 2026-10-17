"""Whisper's network and its hearing, on a tiny model with random weights on the processor.

What the real model does was measured against transformers 5.8 on fourteen
songs (2026-10-01): every greedy window the same, token for token. Here the
shapes are cut down so the same code runs in a second without the weights.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from yue2_comfy.whisper import decode, model, network  # noqa: E402

GENERATION = {
    "decoder_start_token_id": 41, "eos_token_id": 40, "no_timestamps_token_id": 47,
    "task_to_id": {"transcribe": 45, "translate": 44}, "max_length": 16,
    "suppress_tokens": [1, 2], "begin_suppress_tokens": [3, 40], "max_initial_timestamp_index": 5,
    "lang_to_id": {"<|en|>": 42, "<|ru|>": 43},
}
BEGIN = 48


@pytest.fixture
def tiny(monkeypatch):
    """A network of the release's layout in miniature, and the vocabulary that fits it."""
    for name, value in (("WIDTH", 16), ("HEADS", 2), ("HEAD", 8), ("ENCODER_LAYERS", 1), ("DECODER_LAYERS", 2),
                        ("FFN", 32), ("CONTEXT", 16), ("VOCAB", 64)):
        monkeypatch.setattr(network, name, value)
    torch.manual_seed(7)
    net = network.Network().eval().requires_grad_(False)
    return net, decode.Vocabulary(GENERATION, 64)


def window_frames(seed=1):
    return torch.randn(network.MELS, decode.WINDOW_FRAMES, generator=torch.Generator().manual_seed(seed))


@torch.inference_mode()
def test_a_prefill_is_the_same_as_its_tokens_fed_one_by_one(tiny):
    net, _vocab = tiny
    static = net.static()
    net.listen(static, net.encode(window_frames()))
    ids = [41, 43, 45, 50, 9]
    whole = net.prefill(ids, static)
    single = [net.prefill(ids[:1], static)[0]]
    for position, token in enumerate(ids[1:], start=1):
        single.append(net.step(token, position, static)[0])
    assert torch.allclose(whole, torch.stack(single), atol=1e-5)


def test_the_cache_starts_as_zeros_and_is_kept_until_let_go(tiny):
    net, _vocab = tiny
    static = net.static()
    assert all(not keys.any() and not values.any() for keys, values in static["cache"] + static["cross"])
    assert net.static() is static
    net.forget_steps()
    assert net.static() is not static


def test_a_window_cannot_open_with_words_or_late(tiny):
    _net, vocab = tiny
    picker = model.Picker(vocab, "cpu")
    scores = picker.scores(torch.zeros(64), [])
    allowed = torch.isfinite(scores[0]).nonzero().flatten().tolist()
    assert allowed == list(range(BEGIN, BEGIN + 6))


def test_after_a_lone_timestamp_come_words_and_after_words_a_timestamp_may_close():
    vocab = decode.Vocabulary(GENERATION, 64)
    picker = model.Picker(vocab, "cpu")
    logits = torch.zeros(64)
    logits[40] = 5.0
    allowed = lambda sampled: torch.isfinite(picker.scores(logits, sampled)[0]).nonzero().flatten().tolist()  # noqa: E731
    after_opening = allowed([BEGIN + 2])
    assert all(token < BEGIN for token in after_opening) and 1 not in after_opening and 40 in after_opening
    assert allowed([BEGIN + 2, 9, BEGIN + 4]) == list(range(40, 47)) + list(range(BEGIN + 4, 64))


def test_a_timestamp_likelier_than_any_word_is_taken():
    vocab = decode.Vocabulary(GENERATION, 64)
    picker = model.Picker(vocab, "cpu")
    logits = torch.zeros(64)
    logits[9] = 3.0
    logits[BEGIN + 4:] = 2.0
    scores = picker.scores(logits, [BEGIN, 9])
    assert not torch.isfinite(scores[0, :BEGIN]).any()
    assert picker.pick(scores, 0.0)[0] >= BEGIN


def test_a_sampled_pick_follows_its_draw_and_never_lands_on_a_banned_token():
    vocab = decode.Vocabulary(GENERATION, 64)
    picker = model.Picker(vocab, "cpu")
    scores = picker.scores(torch.linspace(0.0, 1.0, 64), [BEGIN, 9])
    for draw in (0.0, 0.3, 0.999999):
        token, logprob = picker.pick(scores, 0.6, torch.tensor(draw))
        assert torch.isfinite(scores[0, token]) and logprob <= 0.0
        assert picker.pick(scores, 0.6, torch.tensor(draw))[0] == token


class Words:
    def decode(self, ids):
        return " ".join(str(token) for token in ids if token < BEGIN)


@torch.inference_mode()
def test_a_song_is_heard_window_by_window_and_the_same_way_every_time(tiny):
    net, vocab = tiny
    audio = 0.1 * torch.randn(70 * decode.SAMPLE_RATE, generator=torch.Generator().manual_seed(3))
    shares = []
    first = model.hear(net, vocab, Words(), audio, "ru", progress=shares.append)
    again = model.hear(net, vocab, Words(), audio, "ru")
    assert first["text"] == again["text"] and first["segments"] == again["segments"]
    assert first["windows"] >= 3 and shares[-1] == 1.0
    starts = [segment["start"] for segment in first["segments"]]
    assert starts == sorted(starts)
    assert not first["disagreed"]


@torch.inference_mode()
def test_a_captured_step_that_soils_the_cache_is_caught_and_the_plain_answer_stands(tiny, monkeypatch):
    """The replayed step picks as the plain one does but leaves garbage in the cache; the first check finds it."""
    net, vocab = tiny
    picker = model.Picker(vocab, "cpu")
    static = net.static()
    net.listen(static, net.encode(window_frames()))
    prompt = vocab.request("ru")
    rows = net.prefill(prompt, static)
    heard = {"disagreed": False}
    plain, _ = model._decode(net, static, picker, rows[-1:], len(prompt), 13, 0.0, None, False, None, heard)

    def soiled(token, position):
        logits = net.step(token, position, static)
        static["cache"][0][0][:, :, position] = 1e4
        return logits

    soiled.reset = lambda start: None
    monkeypatch.setattr(net, "stepper", lambda static, start: soiled)
    caught, _ = model._decode(net, static, picker, rows[-1:], len(prompt), 13, 0.0, None, True, None, heard)
    assert caught == plain and heard["disagreed"] and static["distrusted"]


@torch.inference_mode()
def test_cancelling_stops_between_windows(tiny):
    net, vocab = tiny
    with pytest.raises(InterruptedError):
        model.hear(net, vocab, Words(), torch.zeros(40 * decode.SAMPLE_RATE), "en", cancelled=lambda: True)


@torch.inference_mode()
def test_the_language_is_asked_of_every_thirty_seconds(tiny):
    net, vocab = tiny
    audio = 0.1 * torch.randn(65 * decode.SAMPLE_RATE, generator=torch.Generator().manual_seed(4))
    windows = model.identify(net, vocab, audio)
    assert len(windows) == 3 and all(set(window) == {"en", "ru"} for window in windows)
    for window in windows:
        assert abs(sum(torch.tensor(list(window.values())).exp().tolist()) - 1.0) < 1e-5


def test_the_weights_load_under_their_own_names_and_nothing_else_fits(tiny, tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    net, _vocab = tiny
    state = {"model." + name: tensor.contiguous() for name, tensor in net.state_dict().items()}
    safetensors.save_file(state, os.path.join(tmp_path, "model.safetensors"))
    loaded = model.load(str(tmp_path), "cpu")
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded.decoder.embed_tokens.weight, net.decoder.embed_tokens.weight)
    state["proj_out.weight"] = state["model.decoder.embed_tokens.weight"].clone()
    other = tmp_path / "other"
    other.mkdir()
    safetensors.save_file(state, os.path.join(other, "model.safetensors"))
    with pytest.raises(ValueError, match="proj_out.weight"):
        model.load(str(other), "cpu")


def test_the_settings_whisper_hears_with_are_put_back(monkeypatch):
    torch.backends.cudnn.benchmark = True
    precision = torch.get_float32_matmul_precision()
    try:
        with network.steady():
            assert torch.backends.cudnn.benchmark is False
            assert torch.get_float32_matmul_precision() == "highest"
        assert torch.backends.cudnn.benchmark is True
        assert torch.get_float32_matmul_precision() == precision
    finally:
        torch.backends.cudnn.benchmark = False


def test_a_vocabulary_is_read_from_the_folder(tmp_path):
    import json

    (tmp_path / "generation_config.json").write_text(json.dumps(GENERATION), encoding="utf-8")
    vocab = model.vocabulary(str(tmp_path))
    assert vocab.timestamp_begin == BEGIN and vocab.size == network.VOCAB
