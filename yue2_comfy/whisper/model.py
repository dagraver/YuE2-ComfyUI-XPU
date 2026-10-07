"""Loading Whisper's weights, and hearing a song with it: its words in a language it is told, and its language.

The weights are the release's ``model.safetensors``, run in float16 on a card
and in float32 on the processor, beside its ``tokenizer.json`` and
``generation_config.json``. The log-mel is the pack's own
(``asr.network.log_mel``): Qwen3-ASR hears through Whisper's feature
extractor, and the two came out the same to the bit on three songs.

``hear`` decodes window after window as ``decode`` describes. Each pick goes
through what transformers' Whisper logits processors do, in their order
(``Picker``); greedy at the first temperature, sampled at the rest from the
best ``decode.TOP_K`` tokens with this pack's own generator, seeded by the
window and the temperature, so a song is heard the same way every time.

The decoding step is replayed from a CUDA graph and checked against the
plain step on its first token and every 64th after it (``asr.decode.greedy``,
the same bookkeeping as Qwen3-ASR's): a check that fails takes the answer
back to the last one that held, and the rest of the song is decoded with the
plain step alone.

Measured on 2026-10-01 against transformers 5.8 on the fourteen songs the
second hearing was measured with (four real Russian songs, ten of the
pack's), each told its language, on an RTX 5090:

- every window both decoded greedily from the same point gave the same
  tokens, 78 of 78, and the windows fell back to sampling at the same places;
  the sampled windows differ, the generators being different, and over five
  seeds of the five songs that fall back the mean WER was 0.155 here and
  0.178 there, each song's spread the same;
- the language of every thirty seconds came out the same to the fourth
  decimal, and summed over the song it named all fourteen;
- the captured step gave the plain step's tokens in all 188 decodings, in a
  fresh process and after the voice separator and SheetSage2 had run in it;
- fourteen songs took 8.9 s with the captured step, 17.8 s with the plain
  one and 41.8 s in transformers, at a peak of 1.74 GiB.
"""

from __future__ import annotations

import json
import logging
import os

import torch
import torch.nn.functional as F

from ..asr import decode as steps
from ..asr import network as listening
from ..asr.model import mono_16k
from ..constants import WHISPER_GENERATION_NAME
from . import decode, network

log = logging.getLogger(__name__)

PREFIX = "model."
SEED = 0
"""Where the sampled temperatures' draws begin; with the window and the temperature it seeds each one."""


def load(folder: str, device) -> network.Network:
    """The network with the folder's weights, on ``device``: float16 on a card, float32 on the processor."""
    from safetensors import safe_open

    device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    with torch.device("meta"):
        net = network.Network()
    state = {}
    with safe_open(os.path.join(folder, "model.safetensors"), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if not name.startswith(PREFIX):
                raise ValueError("Whisper's weights do not fit: unexpected tensor '{}'".format(name))
            state[name[len(PREFIX):]] = handle.get_tensor(name).to(dtype)
    missing, unexpected = net.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError("Whisper's weights do not fit: missing {} unexpected {}".format(missing[:5], unexpected[:5]))
    return net.to(device).eval().requires_grad_(False)


def vocabulary(folder: str) -> decode.Vocabulary:
    """The ids the release speaks in, from its generation config."""
    with open(os.path.join(folder, WHISPER_GENERATION_NAME), "r", encoding="utf-8") as handle:
        return decode.Vocabulary(json.load(handle), network.VOCAB)


class Picker:
    """transformers' Whisper logits processors, on the card, and the pick after them.

    Their order there is: the tokens banned at the start, the tokens never
    written, then the timestamp rules (``decode.timestamp_rule``), and last
    the rule that a timestamp whose total chance beats every word's is taken.
    """

    def __init__(self, vocab: decode.Vocabulary, device):
        self.vocab = vocab
        every = torch.arange(vocab.size, device=device)

        def banned(tokens):
            if not tokens:
                return None
            return torch.isin(every, torch.tensor(list(tokens), device=device))[None]

        self.suppress = banned(vocab.suppress)
        self.begin_suppress = banned(vocab.begin_suppress)

    def scores(self, logits: torch.Tensor, sampled) -> torch.Tensor:
        """``[1, vocab]`` float32 scores after the processors, for the pick after ``sampled``, the window's answer so far."""
        vocab = self.vocab
        begin = vocab.timestamp_begin
        minus = float("-inf")
        scores = logits.view(1, -1)
        if not sampled and self.begin_suppress is not None:
            scores = torch.where(self.begin_suppress, minus, scores)
        if self.suppress is not None:
            scores = torch.where(self.suppress, minus, scores)
        scores = scores.clone()
        scores[:, vocab.no_timestamps] = minus
        rule = decode.timestamp_rule(sampled, vocab)
        if rule.text:
            scores[:, :vocab.end] = minus
        if rule.timestamps:
            scores[:, begin:] = minus
        if rule.floor:
            scores[:, begin:rule.floor] = minus
        if rule.first:
            scores[:, :begin] = minus
            if vocab.max_initial is not None:
                scores[:, begin + vocab.max_initial + 1:] = minus
        logprobs = F.log_softmax(scores.float(), dim=-1)
        stamps = logprobs[0, begin:].logsumexp(dim=-1)
        words = logprobs[0, :begin].max()
        only = scores.clone()
        only[:, :begin] = minus
        return torch.where(stamps > words, only, scores)

    def _warped(self, scores, temperature: float):
        warped = scores / temperature
        kth = torch.topk(warped, min(decode.TOP_K, warped.shape[-1]))[0][..., -1, None]
        return warped.masked_fill(warped < kth, float("-inf"))

    def pick(self, scores: torch.Tensor, temperature: float, draw=None) -> tuple:
        """``(token, log-probability)``: the best at temperature 0, else sampled with ``draw``, a uniform in [0, 1).

        The log-probability is the one transformers' threshold reads: of the
        scores after the processors and, when sampled, after the temperature
        and the top-k cut, scaled back by the temperature.
        """
        if temperature > 0:
            warped = self._warped(scores, temperature)
            ladder = F.softmax(warped, dim=-1)[0].cumsum(0)
            target = ((1.0 - draw) * ladder[-1]).view(1)
            token = torch.searchsorted(ladder, target).clamp_(max=ladder.numel() - 1)[0]
            logprob = F.log_softmax((warped * temperature).float(), dim=-1)[0, token]
        else:
            token = torch.argmax(scores, dim=-1)[0]
            logprob = F.log_softmax(scores.float(), dim=-1)[0, token]
        pair = torch.stack((token.to(torch.float64), logprob.to(torch.float64))).tolist()
        return int(pair[0]), pair[1]

    def stands(self, scores: torch.Tensor, fast: int, temperature: float, draw, position: int) -> bool:
        """Whether the plain step's ``scores`` let the replayed step's pick ``fast`` stand.

        Greedy, a pick within a near tie of the best does (``asr.network.near_tie``).
        Sampled with the same draw, the two steps pick alike unless the draw
        falls at a boundary the rounding moved; then any token the plain step
        could have sampled at all does.
        """
        if temperature <= 0:
            return listening.near_tie(scores[0], fast, position, tag="whisper")
        if self.pick(scores, temperature, draw)[0] == fast:
            return True
        return bool(torch.isfinite(self._warped(scores, temperature)[0, fast]))


@torch.inference_mode()
def hear(net: network.Network, vocab: decode.Vocabulary, tokenizer, audio: torch.Tensor, code: str,
         cancelled=None, progress=None) -> dict:
    """The words of ``[samples]`` of mono 16 kHz audio heard in the language ``code``.

    ``code`` may also be several codes: each window is then told the one of
    them Whisper finds likeliest in it (``_likeliest``), for a song sung in
    more than one language.

    ``{"text", "segments", "windows", "languages", "again", "skipped", "disagreed"}``:
    ``languages`` is the code each window was told, ``segments`` are
    ``{"start", "end", "tokens"}`` in seconds; ``again`` counts
    the decodings repeated at a higher temperature, ``skipped`` the windows
    dropped as silence, ``disagreed`` whether the captured step was dropped.
    ``progress(share)`` is told how much of the song is heard.
    """
    if audio.numel() and not torch.isfinite(audio).all():
        raise ValueError("the audio holds samples that are not finite numbers")
    with network.steady():
        return _hear(net, vocab, tokenizer, audio.float(), code, cancelled, progress)


def _hear(net, vocab, tokenizer, audio, code, cancelled, progress):
    if audio.numel() < decode.WINDOW_SAMPLES:
        audio = F.pad(audio, (0, decode.WINDOW_SAMPLES - audio.numel()))
    mel = listening.log_mel(audio)
    total = mel.shape[1]
    codes = [code] if isinstance(code, str) else list(code)
    prompt = vocab.request(codes[0])
    start = len(prompt)
    limit = vocab.context - start
    picker = Picker(vocab, net.device)
    static = net.static()
    graphs = net.device.type == "cuda" and network.GRAPHS
    heard = {"segments": [], "windows": 0, "again": 0, "skipped": 0, "disagreed": False, "languages": []}
    seek = 0
    while seek < total:
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled while Whisper was listening")
        frames = min(total - seek, decode.WINDOW_FRAMES)
        net.listen(static, net.encode(F.pad(mel[:, seek:seek + frames], (0, decode.WINDOW_FRAMES - frames))))
        if len(codes) > 1:
            prompt = vocab.request(_likeliest(net, vocab, static, codes))
        heard["languages"].append(next(c for c, ident in vocab.languages.items() if ident == prompt[1]))
        rows = net.prefill(prompt, static)
        no_speech = float(rows[:1].softmax(dim=-1)[0, vocab.no_speech])
        for tried, temperature in enumerate(decode.TEMPERATURES):
            draws = None
            if temperature > 0:
                generator = torch.Generator().manual_seed(SEED + 1000 * heard["windows"] + tried)
                draws = torch.rand(limit + 1, generator=generator).to(net.device)
            answer, logprobs = _decode(net, static, picker, rows[-1:], start, limit, temperature, draws,
                                       graphs, cancelled, heard)
            again, skip = decode.verdict(answer, logprobs, no_speech, vocab.size)
            if not again or tried == len(decode.TEMPERATURES) - 1:
                break
            heard["again"] += 1
        heard["windows"] += 1
        if skip:
            heard["skipped"] += 1
            seek += frames
        else:
            tokens = answer[:-1] if answer and answer[-1] == vocab.end else answer
            segments, advance = decode.cut(tokens, vocab, seek, frames)
            heard["segments"] += segments
            seek += advance
        if progress is not None:
            progress(min(1.0, seek / total))
    tokens = [token for segment in heard["segments"] for token in segment["tokens"]]
    heard["text"] = tokenizer.decode(tokens).strip()
    return heard


def _likeliest(net, vocab, static, codes) -> str:
    """The code among ``codes`` Whisper finds likeliest after ``<|startoftranscript|>`` in the window it is listening to."""
    logits = net.prefill([vocab.start], static)[0]
    return max(codes, key=lambda code: float(logits[vocab.languages[code]]))


def _decode(net, static, picker, first_logits, start, limit, temperature, draws, graphs, cancelled, heard):
    """One decoding of a window at one temperature: ``(answer, logprobs)``, the answer with its end token when written."""
    fed = {}
    picked = {}

    def draw(position):
        return None if draws is None else draws[position - start]

    def sampled(position):
        return [fed[at] for at in range(start, position + 1)]

    def after(logits, token, position):
        fed[position] = token
        nxt, logprob = picker.pick(picker.scores(logits, sampled(position)), temperature, draw(position + 1))
        picked[position + 1] = logprob
        return nxt

    first, logprob = picker.pick(picker.scores(first_logits, []), temperature, draw(start))
    picked[start] = logprob

    def plain(token, position):
        return after(net.step(token, position, static), token, position)

    stepper = net.stepper(static, start) if graphs and not static.get("distrusted") else None
    replayed = None if stepper is None else (lambda token, position: after(stepper(token, position), token, position))
    checked = [start]

    def check(token, position, fast):
        cache = static["cache"]
        saved = [(keys[:, :, position].clone(), values[:, :, position].clone()) for keys, values in cache]
        logits = net.step(token, position, static)
        if not listening.cache_agrees(cache, saved, checked[0], position, tag="whisper"):
            return False
        checked[0] = position + 1
        return picker.stands(picker.scores(logits, sampled(position)), fast, temperature, draw(position + 1), position)

    def disagreed():
        log.warning("[yue2_comfy.whisper] the captured step disagreed with the plain one; decoding without it")
        static["distrusted"] = True
        heard["disagreed"] = True

    answer = steps.greedy(first, start, limit, (picker.vocab.end,), plain, replayed, cancelled, None, disagreed,
                          check)
    return answer, [picked[start + index] for index in range(len(answer))]


@torch.inference_mode()
def identify(net: network.Network, vocab: decode.Vocabulary, audio: torch.Tensor, cancelled=None,
             progress=None) -> list:
    """``[{code: log-probability}]``, Whisper's chance of every language it knows, one for every thirty seconds.

    Each piece is heard on its own, padded to thirty seconds, and asked for
    the token after ``<|startoftranscript|>`` among the language tokens; the
    measurement summed these over the song (``decode.best_language``).
    """
    if audio.numel() and not torch.isfinite(audio).all():
        raise ValueError("the audio holds samples that are not finite numbers")
    audio = audio.float()
    ids = sorted(vocab.languages.values())
    codes = {ident: code for code, ident in vocab.languages.items()}
    windows = []
    with network.steady():
        static = net.static()
        pieces = decode.identified_pieces(audio.numel())
        for index, (first, last) in enumerate(pieces):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled while Whisper was naming the language")
            piece = F.pad(audio[first:last], (0, decode.WINDOW_SAMPLES - (last - first)))
            net.listen(static, net.encode(listening.log_mel(piece)))
            languages = net.prefill([vocab.start], static)[0][ids]
            shares = (languages - torch.logsumexp(languages, dim=0)).tolist()
            windows.append({codes[ident]: share for ident, share in zip(ids, shares)})
            if progress is not None:
                progress((index + 1) / len(pieces))
    return windows
