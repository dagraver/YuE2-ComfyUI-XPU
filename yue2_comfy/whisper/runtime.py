"""Whisper kept between runs, and what it has heard.

Whisper is a song's second hearing: its words beside Qwen3-ASR's, and its
judgement of the language the song is sung in. Both are deterministic -- the
sampled temperatures draw from the pack's own seeded generator -- so they
are kept: the words by the recording, the language they were heard in and
the weights; the language windows by the recording and the weights.

The model stays loaded only as long as the caller lets it ('keep_model_loaded'),
and the Unload Models button lets it go (see ``memory``). With 'low_vram' the
listeners take turns: Whisper leaves the card before Qwen3-ASR or its word
aligner comes, and they leave before Whisper comes. Whisper is never packed:
its weights are 1.5 GiB in float16, smaller than the packed speech model.
"""

from __future__ import annotations

import collections
import gc
import logging
import os
import threading

log = logging.getLogger(__name__)

KEEP_RESULTS = 8
ROOM_BYTES = int(2.5 * 1024 ** 3)
"""Free VRAM wanted before loading: 1.5 GiB of weights and 1.74 GiB at the peak of a song, measured, and a margin."""

_LOCK = threading.RLock()
_STATE = {"key": None, "net": None, "tokenizer": None, "vocab": None}
_HEARD = collections.OrderedDict()
_IDENTIFIED = collections.OrderedDict()


def stamp(folder: str) -> tuple:
    """The weights file's identity for caching: a rewritten file is a different file."""
    from ..asr.runtime import stamp as weights_stamp

    return weights_stamp(folder)


def is_loaded() -> bool:
    return _STATE["net"] is not None


def unload() -> None:
    """Let go of the network and give its memory back; what it heard stays."""
    with _LOCK:
        net = _STATE["net"]
        _STATE.update(key=None, net=None, tokenizer=None, vocab=None)
    if net is None:
        return
    del net
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            log.debug("[yue2_comfy.whisper] no cache to empty", exc_info=True)


def acquire(folder: str, device, progress=None, low_vram: bool = False) -> tuple:
    """``(net, tokenizer, vocabulary)`` for this folder on this device, loaded when they are not already.

    With ``low_vram`` the speech model and its aligner leave the card first.
    """
    from ..asr import runtime as speech

    key = (stamp(folder), str(device))
    with _LOCK:
        if _STATE["key"] == key and _STATE["net"] is not None:
            return _STATE["net"], _STATE["tokenizer"], _STATE["vocab"]
        unload()
        if low_vram:
            speech.unload()
            speech.unload_aligner()
        speech._make_room(device, ROOM_BYTES, [("Qwen3-ForcedAligner", speech.aligner_loaded, speech.unload_aligner),
                                               ("Qwen3-ASR", speech.is_loaded, speech.unload)])
        if progress is not None:
            progress.text("Loading Whisper", force=True)
        from ..constants import WHISPER_TOKENIZER_NAME
        from . import model
        from .tokenizer import Tokenizer

        tokenizer = Tokenizer(os.path.join(folder, WHISPER_TOKENIZER_NAME))
        vocab = model.vocabulary(folder)
        net = model.load(folder, device)
        _STATE.update(key=key, net=net, tokenizer=tokenizer, vocab=vocab)
        log.info("[yue2_comfy.whisper] loaded %s on %s", os.path.basename(folder), device)
        return net, tokenizer, vocab


def _get(store, key):
    with _LOCK:
        found = store.get(key)
        if found is not None:
            store.move_to_end(key)
        return found


def _put(store, key, value) -> None:
    with _LOCK:
        store[key] = value
        store.move_to_end(key)
        while len(store) > KEEP_RESULTS:
            store.popitem(last=False)


def _forget_steps(net) -> None:
    forget = getattr(net, "forget_steps", None)
    if callable(forget):
        forget()


def hear(folder: str, device, waveform, rate: int, key, code: str, progress=None, cancelled=None,
         low_vram: bool = False) -> dict:
    """What Whisper hears in ``waveform`` told the language ``code``: ``model.hear``'s answer, kept by ``key``.

    ``key`` names the recording. The captured decoding step and its cache are
    let go however the hearing ends. ``progress`` is a node progress, told
    how much of the song is heard.
    """
    full_key = (stamp(folder), str(key), str(code))
    found = _get(_HEARD, full_key)
    if found is not None:
        return found
    net, tokenizer, vocab = acquire(folder, device, progress, low_vram=low_vram)
    from . import model

    told = None if progress is None else (lambda share: progress.ratio(share, "Listening again with Whisper"))
    try:
        heard = model.hear(net, vocab, tokenizer, model.mono_16k(waveform, rate), code, cancelled=cancelled,
                           progress=told)
    finally:
        _forget_steps(net)
    _put(_HEARD, full_key, heard)
    return heard


def identify(folder: str, device, waveform, rate: int, key, progress=None, cancelled=None,
             low_vram: bool = False) -> list:
    """Whisper's language windows for ``waveform`` (``model.identify``), kept by ``key``, the recording."""
    full_key = (stamp(folder), str(key))
    found = _get(_IDENTIFIED, full_key)
    if found is not None:
        return found
    net, _tokenizer, vocab = acquire(folder, device, progress, low_vram=low_vram)
    from . import model

    told = None if progress is None else (lambda share: progress.ratio(share, "Whisper names the language"))
    try:
        windows = model.identify(net, vocab, model.mono_16k(waveform, rate), cancelled=cancelled, progress=told)
    finally:
        _forget_steps(net)
    _put(_IDENTIFIED, full_key, windows)
    return windows
