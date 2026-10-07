"""Whisper's long-form transcription, the parts that need no network. Standard library only.

A song is heard thirty seconds at a time, the way OpenAI's long-form
transcription does it and ``WhisperForConditionalGeneration.generate`` in
transformers 5.8 repeats it, the version the second hearing was measured with:

- each window is asked ``<|startoftranscript|><|xx|><|transcribe|>`` and
  answers in timestamped segments, ``<|t0|> words<|t1|><|t1|> words<|t2|>``;
- the answer is cut at its pairs of timestamps, and the next window starts
  where the last whole segment ended, or after the whole window when the
  answer ends on a single timestamp, which means nothing more is said in it
  (``cut``);
- an answer that repeats itself, so that it compresses too well, or one the
  model was unsure of, its mean log-probability low, is decoded again,
  sampled at the next temperature; one it was unsure of in a window it took
  for silence is dropped (``verdict``).

Which tokens may come next is ``timestamp_rule``: timestamps in pairs, never
going back, the first one within the first second. ``Vocabulary`` reads the
ids from the release's generation config.

The language of a song is the one whose log-probabilities, summed over every
thirty seconds of it, are highest (``best_language``). Measured on 2026-10-01
on 14 songs: that named all 14, while Whisper's own long-form choice takes the
language of the first window alone, and songs that open without a voice were
then heard as a translation or as nonsense.

Thresholds are compared in float32, as transformers compares them, so a mean
log-probability that lands on -1.0 is judged as it was in the measurement.
"""

from __future__ import annotations

import math
import struct
import zlib
from typing import NamedTuple

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 30 * SAMPLE_RATE
WINDOW_FRAMES = 3000
"""Log-mel frames, 100 a second, in the thirty seconds the encoder hears at once."""
INPUT_STRIDE = 2
"""Mel frames to one encoder position; a timestamp step of 0.02 s is two frames."""
TIME_PRECISION = 0.02
SHORTEST_IDENTIFIED = SAMPLE_RATE
"""The last piece of a song shorter than a second is not asked its language."""

TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
COMPRESSION_RATIO = 1.35
LOGPROB = -1.0
NO_SPEECH = 0.6
TOP_K = 50
"""The sampled temperatures pick among this many best tokens: transformers' default, which the measurement ran with."""


class Vocabulary:
    """The ids one Whisper release speaks in, read from its ``generation_config.json``."""

    def __init__(self, generation: dict, size: int):
        self.size = int(size)
        self.start = int(generation["decoder_start_token_id"])
        self.end = int(generation["eos_token_id"])
        self.no_timestamps = int(generation["no_timestamps_token_id"])
        self.timestamp_begin = self.no_timestamps + 1
        self.no_speech = self.no_timestamps - 1
        self.transcribe = int(generation["task_to_id"]["transcribe"])
        self.context = int(generation["max_length"])
        self.suppress = tuple(int(token) for token in generation.get("suppress_tokens") or ())
        self.begin_suppress = tuple(int(token) for token in generation.get("begin_suppress_tokens") or ())
        initial = generation.get("max_initial_timestamp_index")
        self.max_initial = None if initial is None else int(initial)
        self.languages = {token.strip("<|>"): int(ident) for token, ident in
                          (generation.get("lang_to_id") or {}).items()}

    def request(self, code: str) -> list:
        """The tokens a window is asked with: start, the language, transcribe; timestamps follow on their own."""
        if code not in self.languages:
            raise ValueError("Whisper does not know the language code '{}'".format(code))
        return [self.start, self.languages[code], self.transcribe]


class Rule(NamedTuple):
    """What ``timestamp_rule`` bans before the next pick."""

    text: bool
    """Every token below the end token: a timestamp that opens a segment's end must close it."""
    timestamps: bool
    """Every timestamp: two in a row are a pair, and words come next."""
    floor: int
    """Timestamps below this id, so time never goes back; 0 bans none."""
    first: bool
    """Everything but a timestamp within the first ``max_initial`` steps: the first pick of a window."""


def timestamp_rule(sampled, vocab: Vocabulary) -> Rule:
    """The bans on the pick after ``sampled``, the tokens this window has answered so far."""
    begin = vocab.timestamp_begin
    last = len(sampled) >= 1 and sampled[-1] >= begin
    penultimate = len(sampled) < 2 or sampled[-2] >= begin
    stamps = [token for token in sampled if token >= begin]
    floor = 0
    if stamps:
        floor = stamps[-1] if last and not penultimate else stamps[-1] + 1
    return Rule(text=last and not penultimate, timestamps=last and penultimate, floor=floor, first=not sampled)


def compression_ratio(tokens, vocab_size: int) -> float:
    """How well the answer's tokens compress: a repeating answer compresses far better than words do."""
    width = int(math.log2(vocab_size) / 8) + 1
    raw = b"".join(int(token).to_bytes(width, "little") for token in tokens)
    return len(raw) / len(zlib.compress(raw))


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def mean_logprob(values) -> float:
    """The mean of the picks' log-probabilities, summed one by one in float32 as transformers sums them."""
    if not values:
        return 0.0
    total = 0.0
    for value in values:
        total = _f32(total + value)
    return _f32(total / len(values))


def verdict(tokens, logprobs, no_speech: float, vocab_size: int) -> tuple:
    """``(again, skip)`` for one decoding of a window.

    ``tokens`` is the answer with its end token when it wrote one, and
    ``logprobs`` the log-probability of each of them where it was picked.
    ``again`` asks for the window to be decoded at the next temperature;
    ``skip`` drops it, for an answer the model was unsure of in a window it
    took for silence (``no_speech``, the chance of its no-speech token).
    """
    again = compression_ratio(tokens, vocab_size) > COMPRESSION_RATIO
    mean = mean_logprob(logprobs)
    if mean < LOGPROB:
        again = True
        if no_speech > _f32(NO_SPEECH):
            return False, True
    return again, False


def cut(tokens, vocab: Vocabulary, seek: int, frames: int) -> tuple:
    """``(segments, advance)``: a window's answer as ``{"start", "end", "tokens"}`` segments, and how far to seek.

    ``tokens`` is the answer without its end token, ``seek`` the mel frame
    the window began at and ``frames`` how many it held. ``advance`` is in mel
    frames: to the last pair of timestamps, the end of the last whole segment,
    or past the whole window when the answer ended on a single timestamp or
    held no pair at all.
    """
    begin = vocab.timestamp_begin
    offset = seek * TIME_PRECISION / INPUT_STRIDE
    flags = [token >= begin for token in tokens]
    single_ending = flags[-2:] == [False, True]
    pairs = [index + 1 for index in range(len(tokens) - 1) if flags[index] and flags[index + 1]]
    if not pairs:
        stamps = [token for token in tokens if token >= begin]
        end = int(frames * 0.01 / TIME_PRECISION)
        if stamps and stamps[-1] != begin:
            end = stamps[-1] - begin
        return [{"start": offset, "end": offset + end * TIME_PRECISION, "tokens": list(tokens)}], frames
    slices = list(pairs)
    if single_ending:
        slices.append(len(tokens))
    else:
        slices[-1] += 1
    segments = []
    last = 0
    for index, current in enumerate(slices):
        piece = list(tokens[last:current])
        closing = -1 if index < len(slices) - 1 or single_ending else -2
        segments.append({"start": offset + (piece[0] - begin) * TIME_PRECISION,
                         "end": offset + (piece[closing] - begin) * TIME_PRECISION, "tokens": piece})
        last = current
    if single_ending:
        return segments, frames
    advance = (tokens[last - 2] - begin) * INPUT_STRIDE
    return segments, advance if advance > 0 else frames


def identified_pieces(samples: int) -> list:
    """``[(first, last)]`` sample ranges of the thirty-second pieces whose language is asked, the short tail left out."""
    pieces = []
    for first in range(0, max(int(samples), 1), WINDOW_SAMPLES):
        last = min(int(samples), first + WINDOW_SAMPLES)
        if last - first >= SHORTEST_IDENTIFIED:
            pieces.append((first, last))
    return pieces


def best_language(windows, candidates) -> str:
    """The code among ``candidates`` whose log-probabilities summed over ``windows`` are highest, or "".

    ``windows`` is one ``{code: log-probability}`` for every piece heard; the
    first of equal sums wins, in the order ``candidates`` gives.
    """
    known = [code for code in candidates if windows and all(code in window for window in windows)]
    if not known:
        return ""
    return max(known, key=lambda code: sum(window[code] for window in windows))
