"""Whisper's long-form bookkeeping without torch: timestamps, cutting a window, the verdict, the language."""

from __future__ import annotations

import struct

import pytest

from yue2_comfy.whisper import decode

GENERATION = {
    "decoder_start_token_id": 50258, "eos_token_id": 50257, "no_timestamps_token_id": 50364,
    "task_to_id": {"transcribe": 50360, "translate": 50359}, "max_length": 448,
    "suppress_tokens": [1, 2, 7], "begin_suppress_tokens": [220, 50257], "max_initial_timestamp_index": 50,
    "lang_to_id": {"<|en|>": 50259, "<|ru|>": 50263, "<|haw|>": 50352},
}
VOCAB = decode.Vocabulary(GENERATION, 51866)
T = VOCAB.timestamp_begin


def stamp(steps):
    return T + steps


def test_the_release_ids_are_read_from_its_generation_config():
    assert (VOCAB.start, VOCAB.end, VOCAB.timestamp_begin, VOCAB.no_speech) == (50258, 50257, 50365, 50363)
    assert VOCAB.languages == {"en": 50259, "ru": 50263, "haw": 50352}
    assert VOCAB.suppress == (1, 2, 7) and VOCAB.begin_suppress == (220, 50257)
    assert (VOCAB.context, VOCAB.max_initial) == (448, 50)


def test_a_window_is_asked_in_the_language_it_is_told():
    assert VOCAB.request("ru") == [50258, 50263, 50360]
    with pytest.raises(ValueError, match="'xx'"):
        VOCAB.request("xx")


@pytest.mark.parametrize("sampled, rule", [
    ([], decode.Rule(text=False, timestamps=False, floor=0, first=True)),
    ([stamp(0)], decode.Rule(text=False, timestamps=True, floor=stamp(1), first=False)),
    ([stamp(0), 11], decode.Rule(text=False, timestamps=False, floor=stamp(1), first=False)),
    ([stamp(0), 11, stamp(40)], decode.Rule(text=True, timestamps=False, floor=stamp(40), first=False)),
    ([stamp(0), 11, stamp(40), stamp(40)], decode.Rule(text=False, timestamps=True, floor=stamp(41), first=False)),
    ([stamp(0), 11, stamp(40), stamp(40), 12], decode.Rule(text=False, timestamps=False, floor=stamp(41),
                                                           first=False)),
])
def test_timestamps_come_in_pairs_and_never_go_back(sampled, rule):
    assert decode.timestamp_rule(sampled, VOCAB) == rule


def test_a_repeating_answer_compresses_far_better_than_words():
    assert decode.compression_ratio([400, 401] * 40, 51866) > decode.COMPRESSION_RATIO
    assert decode.compression_ratio(list(range(1000, 1060)), 51866) < decode.COMPRESSION_RATIO
    assert decode.compression_ratio([], 51866) == 0.0


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def test_the_mean_is_summed_in_float32_one_pick_at_a_time():
    values = [f32(-0.1), f32(-0.2), f32(-0.3)]
    total = f32(f32(values[0] + values[1]) + values[2])
    assert decode.mean_logprob(values) == f32(total / 3)
    assert decode.mean_logprob([]) == 0.0


def test_a_repeating_answer_is_heard_again():
    assert decode.verdict([400, 401] * 40, [-0.1] * 80, 0.0, 51866) == (True, False)


def test_an_unsure_answer_is_heard_again_unless_the_window_is_silence():
    tokens = list(range(1000, 1010))
    assert decode.verdict(tokens, [-1.5] * 10, 0.1, 51866) == (True, False)
    assert decode.verdict(tokens, [-1.5] * 10, 0.9, 51866) == (False, True)


def test_the_thresholds_are_compared_as_transformers_compares_them():
    """A mean of exactly -1.0 is not below it, and a no-speech chance of 0.6 in float32 is not above it."""
    tokens = list(range(1000, 1010))
    assert decode.verdict(tokens, [-1.0] * 10, 0.9, 51866) == (False, False)
    assert decode.verdict(tokens, [-1.5] * 10, f32(0.6), 51866) == (True, False)


def test_a_window_ending_on_one_timestamp_is_cut_at_its_pairs_and_heard_to_its_end():
    tokens = [stamp(0), 11, 12, stamp(100), stamp(100), 13, stamp(200)]
    segments, advance = decode.cut(tokens, VOCAB, 0, 3000)
    assert [(s["start"], s["end"], s["tokens"]) for s in segments] == [
        (0.0, 2.0, [stamp(0), 11, 12, stamp(100)]), (2.0, 4.0, [stamp(100), 13, stamp(200)])]
    assert advance == 3000


def test_a_window_cut_off_mid_segment_goes_on_from_its_last_pair():
    tokens = [stamp(0), 11, stamp(100), stamp(100), 12, stamp(150), stamp(150), 13]
    segments, advance = decode.cut(tokens[:-1], VOCAB, 3000, 3000)
    assert [(s["start"], s["end"]) for s in segments] == [(30.0, 32.0), (32.0, 33.0)]
    assert segments[-1]["tokens"] == [stamp(100), 12, stamp(150), stamp(150)]
    assert advance == 300


def test_a_window_without_a_pair_is_one_segment_to_its_last_timestamp():
    segments, advance = decode.cut([stamp(0), 11, 12, stamp(50)], VOCAB, 0, 3000)
    assert [(s["start"], s["end"]) for s in segments] == [(0.0, 1.0)] and advance == 3000
    segments, advance = decode.cut([stamp(0), 11, 12], VOCAB, 0, 2000)
    assert [(s["start"], s["end"]) for s in segments] == [(0.0, 20.0)] and advance == 2000
    segments, advance = decode.cut([], VOCAB, 0, 3000)
    assert segments[0]["tokens"] == [] and advance == 3000


def test_a_pair_at_the_very_start_does_not_hold_the_song_in_place():
    segments, advance = decode.cut([stamp(0), stamp(0)], VOCAB, 600, 3000)
    assert advance == 3000


def test_the_language_is_asked_of_every_thirty_seconds_but_a_short_tail():
    second = decode.SAMPLE_RATE
    assert decode.identified_pieces(70 * second) == [(0, 30 * second), (30 * second, 60 * second),
                                                     (60 * second, 70 * second)]
    assert decode.identified_pieces(60 * second + second // 2) == [(0, 30 * second), (30 * second, 60 * second)]
    assert decode.identified_pieces(second // 2) == []


def test_the_language_is_the_one_most_likely_over_the_whole_song():
    windows = [{"ru": -0.1, "en": -2.0, "uk": -3.0}, {"ru": -3.0, "en": -0.5, "uk": -3.0}]
    assert decode.best_language(windows, ["ru", "en", "uk"]) == "en"
    assert decode.best_language(windows, ["ru", "uk"]) == "ru"
    assert decode.best_language([{"ru": -1.0, "en": -1.0}], ["en", "ru"]) == "en"
    assert decode.best_language(windows, ["xx"]) == "" and decode.best_language([], ["ru"]) == ""
