"""Choosing which GPU the song is generated on.

One spelling, and it has to survive a saved workflow moving between machines:

===========  =================================================================
``auto``     whatever ComfyUI itself is using
``cuda:N``   a specific CUDA ordinal
``xpu:N``    a specific Intel ordinal
``cpu``      no GPU at all; correct, and about an hour per song
===========  =================================================================

The values are deliberately plain. A label carrying the card's model name would
read better and would break every saved workflow the day the card is replaced.

Both accelerators are asked the same questions of the module their kind names.
torch spells CUDA and XPU identically -- ``is_available``, ``device_count``,
``get_device_name``, ``get_device_properties``, ``mem_get_info``,
``memory_allocated``, ``memory_reserved``, ``empty_cache`` and ``synchronize``
all take the same arguments on ``torch.cuda`` and ``torch.xpu`` -- so nothing
below branches on the kind except where the answer genuinely differs. Exactly
one of the two modules is ever usable in one build: ``torch.xpu`` is absent
before Intel support was compiled in and ``torch.cuda`` answers False in an
Intel build.

Kept in this module rather than a new one because every caller already imports
it, and because "which GPU" and "how much room is on it" are the same question
asked twice.

Ported from the MiniMax-H3 Prompt Rewriter pack, minus the llama.cpp helpers.
torch is imported inside the functions so this module is safe to import while
ComfyUI is still building its node list.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

AUTO = "auto"
CPU = "cpu"
CUDA, XPU = "cuda", "xpu"

ACCELERATORS = (CUDA, XPU)
"""The accelerator kinds this pack can name outright, in the order they are offered.

NVIDIA first because a machine that has one is the machine every earlier
workflow was saved on; an Intel build simply has no CUDA devices to list.
ComfyUI spells its own switch the same way (``--cuda-device``, ``--xpu-device``).
"""

PREFIX = CUDA + ":"
"""The one accelerator spelling that predates Intel support, kept for the callers
that still say it and for the messages written before there were two."""


def _torch():
    import torch

    return torch


def backend(kind: str):
    """The torch module an accelerator kind names, or None when this build has none.

    Presence is not availability: ``torch.cuda`` exists in an Intel build and
    answers ``is_available()`` False. None also covers a process where torch
    itself cannot be imported yet -- ComfyUI reads this module while it is still
    building its node list, and a choice list is not the place to fail.
    """
    kind = (kind or "").strip().lower()
    if kind not in ACCELERATORS:
        return None
    try:
        return getattr(_torch(), kind, None)
    except Exception:
        log.debug("[yue2_comfy.devices.backend] torch has no %s to ask", kind, exc_info=True)
        return None


def spelling(kind: str, ordinal: int) -> str:
    """The saved-workflow spelling of one device: ``cuda:0``, ``xpu:1``."""
    return "{}:{}".format(kind, int(ordinal))


def count(kind: str = CUDA) -> int:
    """How many devices of one accelerator kind this process can see.

    Note *this process*: ComfyUI started with --cuda-device 1 sets
    CUDA_VISIBLE_DEVICES, so there is exactly one visible device and it is
    numbered 0. An Intel build narrows the same way through ZE_AFFINITY_MASK.
    """
    module = backend(kind)
    try:
        return int(module.device_count()) if module is not None and module.is_available() else 0
    except Exception:
        log.debug("[yue2_comfy.devices.count] no %s visible", kind, exc_info=True)
        return 0


def counts() -> dict:
    """Every accelerator kind and how many devices of it are visible."""
    return {kind: count(kind) for kind in ACCELERATORS}


def kinds() -> list:
    """The accelerator kinds with at least one device visible, in offer order."""
    return [kind for kind in ACCELERATORS if count(kind)]


def describe(kind, ordinal=None) -> str:
    """"NVIDIA GeForce RTX 5090, 32.0 GB", or "" when it cannot be read.

    ``describe(0)`` is the CUDA spelling this had before Intel cards mattered;
    ``describe("xpu", 0)`` asks the Intel module instead.
    """
    if ordinal is None:
        kind, ordinal = CUDA, kind
    module = backend(kind)
    if module is None:
        return ""
    try:
        name = module.get_device_name(ordinal)
        total = module.get_device_properties(ordinal).total_memory / 1024 ** 3
        return "{}, {:.1f} GB".format(name, total)
    except Exception:
        log.debug("[yue2_comfy.devices.describe] %s:%s unreadable", kind, ordinal, exc_info=True)
        return ""


_CHOICES = None
_TOOLTIP = None


def choices() -> list:
    """The values offered by the options node, in a stable spelling."""
    global _CHOICES
    if _CHOICES is None:
        _CHOICES = [AUTO]
        for kind in ACCELERATORS:
            _CHOICES += [spelling(kind, ordinal) for ordinal in range(count(kind))]
        _CHOICES.append(CPU)
    return list(_CHOICES)


def tooltip() -> str:
    global _TOOLTIP
    if _TOOLTIP is not None:
        return _TOOLTIP
    lines = [
        "Which device generates the song. 'auto' follows ComfyUI.",
        "",
        "Pick a second card and two things change: the model no longer competes "
        "with ComfyUI's own for VRAM, and 'keep_model_loaded' becomes worth "
        "turning on, because nothing has to be evicted to make room.",
        "",
        "Two YuE2 nodes with different devices in one graph will reload 6.8 GB on "
        "every run, because the loaded model is cached per device.",
        "",
        "'cpu' works and takes roughly an hour per song.",
    ]
    found = ["  " + spelling(kind, ordinal) + " -- " + (describe(kind, ordinal) or "unreadable")
             for kind in ACCELERATORS for ordinal in range(count(kind))]
    if found:
        lines += ["", "Visible here:"] + found
    else:
        lines += ["", "No CUDA or XPU device is visible to ComfyUI, so only 'cpu' will do anything."]
    _TOOLTIP = "\n".join(lines)
    return _TOOLTIP


def parse(spec: str):
    """The ``(kind, ordinal)`` a device spec names, or ``(None, None)`` for 'auto', 'cpu' and nonsense."""
    text = (spec or AUTO).strip().lower()
    for kind in ACCELERATORS:
        prefix = kind + ":"
        if text.startswith(prefix):
            try:
                return kind, int(text[len(prefix):])
            except ValueError:
                return None, None
    return None, None


def index(spec: str):
    """The ordinal of an accelerator spec, or None for 'auto' and 'cpu'.

    Kept for callers that speak of one card at a time; llama.cpp's ``main_gpu``
    and ``--device CUDA{N}`` are the last of them, and an Intel ordinal is
    passed the same way.
    """
    return parse(spec)[1]


def kind(spec: str):
    """The accelerator kind a spec names, or None for 'auto' and 'cpu'."""
    return parse(spec)[0]


def is_cpu(spec: str) -> bool:
    return (spec or AUTO).strip().lower() == CPU


CPU_NOTICE = ("This run is on the CPU, which takes about an hour for a song. "
              "Point 'device' at a CUDA or XPU card if this machine has one.")


def cpu_notice(spec: str):
    """What to say when a run is about to happen on the CPU, or None when it is not.

    'auto' is the spelling that surprises people: on a machine whose accelerator
    is missing or broken it resolves to the CPU, and the run then takes an hour
    without ever saying why. Refusing instead would be worse -- upstream
    supports the CPU, and a mode somebody deliberately chose should run -- so
    the run goes ahead and says what it is doing. The tooltip carries the same
    sentence, but a tooltip is read before the choice, not after it.
    """
    try:
        device = resolve(spec)
    except Exception:
        log.debug("[yue2_comfy.devices.cpu_notice] device unreadable", exc_info=True)
        return None
    return CPU_NOTICE if getattr(device, "type", None) == CPU else None


def validate(spec: str) -> str:
    """Return the spec, refusing one this machine cannot honour.

    Refused rather than quietly demoted to 'auto': a workflow moved from a
    two-card machine asks for a card that is not there, and silently running on
    the wrong one is how somebody's video model gets evicted mid-batch.
    """
    spec = (spec or AUTO).strip() or AUTO
    kind, ordinal = parse(spec)
    if kind is None:
        if spec.lower() in (AUTO, CPU):
            return spec.lower()
        raise ValueError(
            "'{}' is not a device. Use '{}', '{}', '{}N' for a CUDA card or '{}N' for an "
            "Intel one.".format(spec, AUTO, CPU, CUDA, XPU))
    visible = count(kind)
    if ordinal < 0 or ordinal >= visible:
        names = ", ".join(spelling(kind, number) for number in range(visible)) or "none"
        raise RuntimeError(
            "This workflow asks for '{}', but ComfyUI can see {} {} device{} ({}). "
            "Pick one that exists, or '{}'.".format(
                spec, visible if visible else "no", kind.upper(),
                "" if visible == 1 else "s", names, AUTO))
    return spec.lower()


def resolve(spec: str):
    """The torch.device a run should use, with 'auto' asking ComfyUI."""
    torch = _torch()
    kind, ordinal = parse(spec)
    if (spec or AUTO).strip().lower() == CPU:
        return torch.device("cpu")
    if kind is not None:
        return torch.device(kind, ordinal)
    try:
        import comfy.model_management as mm

        return mm.get_torch_device()
    except Exception:
        log.debug("[yue2_comfy.devices.resolve] no comfy device", exc_info=True)
    for candidate in ACCELERATORS:
        if count(candidate):
            try:
                return torch.device(candidate, backend(candidate).current_device())
            except Exception:
                log.debug("[yue2_comfy.devices.resolve] %s unreadable", candidate, exc_info=True)
    return torch.device("cpu")


def llama_arguments(spec: str) -> list:
    """The device part of an llama.cpp command line.

    'auto' adds nothing at all, which lets llama.cpp make its own choice -- it
    is the only one of the four that knows whether this build has a Vulkan
    device, a CUDA one, a SYCL one or neither.

    A CUDA ordinal is spelled out, because llama.cpp names those ``CUDA0``,
    ``CUDA1``. An Intel one is not: llama.cpp has no XPU backend to name, the
    card is reached through its Vulkan or SYCL build, and those spell and order
    their devices differently (``Vulkan0``, ``SYCL0``) from the way torch does.
    Passing an index would pick a card by a numbering the build makes up; the
    offload itself still happens through ``layers_for``.
    """
    if is_cpu(spec):
        return ["--device", "none"]
    kind, ordinal = parse(spec)
    return ["--device", "CUDA{}".format(ordinal)] if kind == CUDA else []


def layers_for(spec: str, gpu_layers: int) -> int:
    """The offload count once the device has had its say."""
    return 0 if is_cpu(spec) else int(gpu_layers)


def shares_comfy_device(spec: str) -> bool:
    """Would this run compete with ComfyUI's own models for VRAM?

    True for 'auto' -- the safe assumption. False only when the answer is
    definitely no, so an unreadable ComfyUI device leaves the eviction in place
    rather than skipping it on a guess. Both kinds are compared, so a run pinned
    to 'xpu:0' is recognized as sharing the card ComfyUI is already using.
    """
    kind, ordinal = parse(spec)
    if kind is None:
        return not is_cpu(spec)
    try:
        import comfy.model_management as mm

        current = mm.get_torch_device()
    except Exception:
        log.debug("[yue2_comfy.devices.shares_comfy_device] no comfy device", exc_info=True)
        return True
    if getattr(current, "type", None) != kind:
        return False
    return (getattr(current, "index", None) or 0) == ordinal


def kind_of(device) -> str:
    """The accelerator kind a torch.device, or its name, belongs to; 'cpu' when it is neither."""
    kind = getattr(device, "type", None)
    if kind is None and isinstance(device, str):
        kind = device.split(":", 1)[0]
    return (kind or CPU).strip().lower()


def module_of(device):
    """The torch module a device belongs to, or None when it is not an accelerator."""
    return backend(kind_of(device))


def is_accelerator(device) -> bool:
    """Whether this is a device of an accelerator kind, rather than the processor.

    Answers for CUDA and XPU alike, which is what the callers that used to write
    ``device.type == "cuda"`` before emptying a cache or reading free memory
    meant all along. Read from the device's own name, not from torch: the
    question is which kind of device this is, and a caller that asks it while
    torch cannot yet be imported -- or in a test process that has none -- still
    gets the true answer. Every operation built on it no-ops safely when there is
    no module to ask.
    """
    return kind_of(device) in ACCELERATORS


def empty_cache(device=None) -> None:
    """Give an accelerator's cached blocks back, for one kind or for every kind.

    Both modules hold memory the driver has already been told is free, so both
    answer ``empty_cache()`` with no arguments. A device of None is the
    process-wide call the loader makes when a model goes.
    """
    for kind in (ACCELERATORS if device is None else (kind_of(device),)):
        module = backend(kind)
        if module is None:
            continue
        try:
            module.empty_cache()
        except Exception:
            log.debug("[yue2_comfy.devices.empty_cache] %s did nothing", kind, exc_info=True)


def synchronize(device) -> None:
    """Wait for the work queued on one accelerator device, where the backend can wait."""
    module = module_of(device)
    if module is None:
        return
    try:
        module.synchronize(device)
    except Exception:
        log.debug("[yue2_comfy.devices.synchronize] %s did nothing", kind_of(device), exc_info=True)


def free_bytes(device):
    """Bytes this process could still allocate on ``device``, or None off an accelerator.

    The driver's free memory plus what the caching allocator holds unused, and
    no more than a per-process memory fraction allows when one is set. The
    arithmetic is the same on both accelerators, which is why it lives here
    rather than beside any one caller.
    """
    module = module_of(device)
    if module is None:
        return None
    try:
        free, total = module.mem_get_info(device)
        allocated = module.memory_allocated(device)
        available = free + module.memory_reserved(device) - allocated
        getter = getattr(getattr(module, "memory", None), "get_per_process_memory_fraction", None)
        if getter is None:
            getter = getattr(module, "get_per_process_memory_fraction", None)
        if getter is not None:
            fraction = float(getter(device))
            if fraction < 1.0:
                available = min(available, int(fraction * total) - allocated)
        return max(0, int(available))
    except Exception:
        log.debug("[yue2_comfy.devices.free_bytes] cannot read free memory", exc_info=True)
        return None


def memory_allocated(device):
    """Bytes torch holds allocated on ``device``, or None when it cannot be read."""
    module = module_of(device)
    if module is None:
        return None
    try:
        return int(module.memory_allocated(device))
    except Exception:
        log.debug("[yue2_comfy.devices.memory_allocated] cannot read it", exc_info=True)
        return None


_OOM = None


def out_of_memory() -> tuple:
    """The exceptions a backend raises when it cannot allocate, as a tuple to catch.

    ``torch.OutOfMemoryError`` is the class both accelerators raise from torch
    2.7 on -- the XPU allocator throws c10's own, not a CUDA one -- and
    ``torch.cuda.OutOfMemoryError`` is kept beside it for older CUDA builds and
    for code that catches it by name. RuntimeError is the last resort of a torch
    that predates both, where an out-of-memory was an ordinary RuntimeError.
    """
    global _OOM
    if _OOM is None:
        torch = _torch()
        found = []
        for candidate in (getattr(torch, "OutOfMemoryError", None),
                          getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),
                          getattr(getattr(torch, "xpu", None), "OutOfMemoryError", None)):
            if isinstance(candidate, type) and issubclass(candidate, BaseException) \
                    and candidate not in found:
                found.append(candidate)
        _OOM = tuple(found) or (RuntimeError,)
    return _OOM
