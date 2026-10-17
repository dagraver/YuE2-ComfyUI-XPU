"""Choosing a device when the card may be Intel's.

The pack asks torch the same questions of ``torch.cuda`` and ``torch.xpu`` --
both answer ``is_available``, ``device_count``, ``get_device_name``,
``get_memory_info``, ``empty_cache`` and the rest with the same arguments -- so
what is checked here is that every question is asked of the module the kind
names, that a spec survives a round trip through a saved workflow, and that a
machine with no accelerator still offers the processor.

A stand-in card is installed as ``devices.backend``'s answer for ``xpu``, which
is how an Intel build is stood in for on a machine that has no Intel card.
"""

from __future__ import annotations

import sys
import types

import pytest

from yue2_comfy import devices

GIB = 1024 ** 3


class Card:
    """A stand-in torch.cuda or torch.xpu, answering only what this pack asks."""

    def __init__(self, names=("Intel Arc B580",), free=6 * GIB, total=12 * GIB,
                 allocated=1 * GIB, reserved=2 * GIB, fraction=1.0, available=True):
        self.names = list(names)
        self.available = available
        self.free, self.total = free, total
        self.allocated, self.reserved = allocated, reserved
        self.fraction = fraction
        self.calls = []
        self.memory = types.SimpleNamespace(get_per_process_memory_fraction=self._fraction)

    def _fraction(self, device):
        return self.fraction

    def is_available(self):
        return self.available

    def device_count(self):
        return len(self.names) if self.available else 0

    def get_device_name(self, ordinal):
        return self.names[ordinal]

    def get_device_properties(self, ordinal):
        return types.SimpleNamespace(total_memory=self.total)

    def current_device(self):
        return 0

    def mem_get_info(self, device):
        return self.free, self.total

    def memory_allocated(self, device):
        return self.allocated

    def memory_reserved(self, device):
        return self.reserved

    def empty_cache(self):
        self.calls.append("empty_cache")

    def synchronize(self, device):
        self.calls.append("synchronize")


@pytest.fixture
def intel(monkeypatch):
    """An Intel build: one Arc card is visible and CUDA is not there at all."""
    card = Card()
    monkeypatch.setattr(devices, "backend", lambda kind: card if kind == "xpu" else None)
    monkeypatch.setattr(devices, "_CHOICES", None)
    monkeypatch.setattr(devices, "_TOOLTIP", None)
    return card


@pytest.fixture
def no_comfy(monkeypatch):
    """A process outside ComfyUI, so 'auto' has to decide for itself."""
    monkeypatch.setitem(sys.modules, "comfy", None)


@pytest.fixture
def fake_torch(monkeypatch):
    """torch's device() alone, so a spec can be resolved on a build without that accelerator."""
    namespace = types.SimpleNamespace(
        device=lambda kind, index=None: types.SimpleNamespace(type=kind, index=index))
    monkeypatch.setattr(devices, "_torch", lambda: namespace)
    return namespace


def test_an_intel_card_is_offered_beside_the_processor(intel):
    assert devices.choices() == ["auto", "xpu:0", "cpu"]
    assert devices.kinds() == ["xpu"]
    assert devices.counts() == {"cuda": 0, "xpu": 1}
    assert devices.count("xpu") == 1


def test_two_intel_cards_are_two_choices(monkeypatch):
    card = Card(names=("Intel Arc B580", "Intel Arc A770"))
    monkeypatch.setattr(devices, "backend", lambda kind: card if kind == "xpu" else None)
    monkeypatch.setattr(devices, "_CHOICES", None)
    assert devices.choices() == ["auto", "xpu:0", "xpu:1", "cpu"]


def test_the_tooltip_names_the_intel_card(intel):
    said = devices.tooltip()
    assert "xpu:0 -- Intel Arc B580, 12.0 GB" in said
    assert "cpu" in said


def test_nothing_visible_still_offers_the_processor(monkeypatch):
    monkeypatch.setattr(devices, "backend", lambda kind: None)
    monkeypatch.setattr(devices, "_CHOICES", None)
    monkeypatch.setattr(devices, "_TOOLTIP", None)
    assert devices.choices() == ["auto", "cpu"]
    assert "No CUDA or XPU device" in devices.tooltip()


def test_a_spec_is_read_as_a_kind_and_an_ordinal():
    assert devices.parse("xpu:1") == ("xpu", 1)
    assert devices.parse("XPU:1") == ("xpu", 1)
    assert devices.parse("cuda:0") == ("cuda", 0)
    assert devices.parse("cuda") == (None, None)
    assert devices.parse("xpu:x") == (None, None)
    assert devices.parse("auto") == (None, None)
    assert devices.parse(None) == (None, None)
    assert devices.kind("xpu:3") == "xpu"
    assert devices.index("xpu:3") == 3
    assert devices.index("cpu") is None


def test_validate_takes_a_visible_intel_card(intel):
    assert devices.validate("xpu:0") == "xpu:0"
    assert devices.validate("XPU:0") == "xpu:0"
    assert devices.validate("auto") == "auto"
    assert devices.validate("CPU") == "cpu"
    assert devices.validate(None) == "auto"


def test_validate_refuses_an_intel_card_that_is_not_there(intel):
    with pytest.raises(RuntimeError) as refused:
        devices.validate("xpu:1")
    assert "XPU" in str(refused.value) and "xpu:0" in str(refused.value)
    with pytest.raises(RuntimeError):
        devices.validate("xpu:-1")


def test_a_stranger_is_still_not_a_device(intel):
    """The message names what this machine does accept, which on an Intel build is the Intel spelling."""
    with pytest.raises(ValueError) as refused:
        devices.validate("gpu:0")
    assert "gpu:0" in str(refused.value)
    assert "xpu" in str(refused.value)


def test_resolve_spells_the_card_the_way_torch_does(intel, no_comfy, fake_torch):
    assert devices.resolve("xpu:1").type == "xpu"
    assert devices.resolve("xpu:1").index == 1
    assert devices.resolve("cuda:0").type == "cuda"
    assert devices.resolve("cpu").type == "cpu"
    assert devices.resolve(None).type == "xpu"


def test_auto_takes_the_intel_card_when_comfy_is_not_there(intel, no_comfy, fake_torch):
    assert devices.resolve("auto").type == "xpu"
    assert devices.resolve("auto").index == 0


def test_auto_is_the_processor_when_no_accelerator_is_visible(monkeypatch, no_comfy, fake_torch):
    monkeypatch.setattr(devices, "backend", lambda kind: None)
    assert devices.resolve("auto").type == "cpu"


def test_the_processor_is_announced_even_on_a_machine_that_has_an_intel_card(intel, no_comfy, fake_torch):
    assert devices.cpu_notice("auto") is None
    assert "about an hour" in devices.cpu_notice("cpu")


def test_free_bytes_reads_an_intel_card_the_same_way(intel):
    """The driver's free memory plus the allocator's unused blocks, less what is allocated."""
    assert devices.free_bytes("xpu:0") == 6 * GIB + 2 * GIB - 1 * GIB
    assert devices.free_bytes("cpu") is None


def test_free_bytes_will_not_exceed_a_memory_fraction(monkeypatch):
    card = Card(fraction=0.25)
    monkeypatch.setattr(devices, "backend", lambda kind: card if kind == "xpu" else None)
    assert devices.free_bytes("xpu:0") == 0.25 * 12 * GIB - 1 * GIB


def test_memory_allocated_is_asked_of_the_card_that_names_it(intel):
    assert devices.memory_allocated("xpu:0") == 1 * GIB
    assert devices.memory_allocated("cpu") is None


def test_empty_cache_asks_whichever_kind_the_device_names(monkeypatch):
    cuda, xpu = Card(), Card()
    monkeypatch.setattr(devices, "backend",
                        lambda kind: {"cuda": cuda, "xpu": xpu}.get(kind))
    devices.empty_cache("xpu:0")
    assert xpu.calls == ["empty_cache"] and cuda.calls == []
    devices.empty_cache()
    assert xpu.calls == ["empty_cache"] * 2 and cuda.calls == ["empty_cache"]


def test_synchronize_does_nothing_on_the_processor(intel):
    devices.synchronize("cpu")
    assert intel.calls == []
    devices.synchronize("xpu:0")
    assert intel.calls == ["synchronize"]


def test_an_intel_card_is_an_accelerator_and_the_processor_is_not(intel):
    assert devices.is_accelerator("xpu:0") is True
    assert devices.is_accelerator("cuda:0") is True
    assert devices.is_accelerator("cpu") is False
    assert devices.is_accelerator("mps") is False
    assert devices.kind_of("XPU:2") == "xpu"
    assert devices.kind_of(types.SimpleNamespace(type="cuda")) == "cuda"
    assert devices.module_of("xpu:0") is intel
    assert devices.module_of("cuda:0") is None


def test_the_kind_of_device_is_known_before_torch_can_be_imported(monkeypatch):
    """The pack is asked this while ComfyUI builds its node list, when torch may not answer yet."""

    def missing():
        raise ModuleNotFoundError("No module named 'torch'")

    monkeypatch.setattr(devices, "_torch", missing)
    assert devices.is_accelerator("xpu:0") is True
    assert devices.is_accelerator("cuda:0") is True
    assert devices.is_accelerator("cpu") is False
    assert devices.count("xpu") == 0
    assert devices.empty_cache("xpu:0") is None
    assert devices.free_bytes("xpu:0") is None


def test_the_llama_command_names_a_cuda_card_and_leaves_an_intel_one_to_its_build():
    """llama.cpp has no XPU backend to name; its Vulkan and SYCL builds number devices their own way."""
    assert devices.llama_arguments("cpu") == ["--device", "none"]
    assert devices.llama_arguments("cuda:1") == ["--device", "CUDA1"]
    assert devices.llama_arguments("xpu:1") == []
    assert devices.llama_arguments("auto") == []
    assert devices.layers_for("xpu:1", -1) == -1
    assert devices.layers_for("cpu", -1) == 0


def test_shares_comfy_device_knows_an_intel_card(monkeypatch):
    management = types.SimpleNamespace(
        get_torch_device=lambda: types.SimpleNamespace(type="xpu", index=0))
    monkeypatch.setitem(sys.modules, "comfy", types.SimpleNamespace(model_management=management))
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    assert devices.shares_comfy_device("xpu:0") is True
    assert devices.shares_comfy_device("xpu:1") is False
    assert devices.shares_comfy_device("cuda:0") is False
    assert devices.shares_comfy_device("cpu") is False
    assert devices.shares_comfy_device("auto") is True


def test_an_out_of_memory_is_the_class_torch_raises_on_either_card():
    """Asked once and held, so the tuple a caller catches is the same one every time."""
    torch = pytest.importorskip("torch")

    caught = devices.out_of_memory()
    assert torch.OutOfMemoryError in caught
    assert isinstance(caught, tuple) and caught
    assert devices.out_of_memory() is caught


def test_describe_still_reads_a_bare_ordinal_as_cuda(monkeypatch):
    cuda = Card(names=("NVIDIA GeForce RTX 5090",), total=32 * GIB)
    monkeypatch.setattr(devices, "backend", lambda kind: cuda if kind == "cuda" else None)
    assert devices.describe(0) == "NVIDIA GeForce RTX 5090, 32.0 GB"
    assert devices.describe("xpu", 0) == ""


def test_both_kinds_are_offered_in_one_list(monkeypatch):
    """NVIDIA first, then Intel, then the processor: a saved workflow keeps the spelling it was saved with."""
    cuda, xpu = Card(names=("RTX 4090",)), Card(names=("Intel Arc B580",))
    monkeypatch.setattr(devices, "backend",
                        lambda kind: {"cuda": cuda, "xpu": xpu}.get(kind))
    monkeypatch.setattr(devices, "_CHOICES", None)
    monkeypatch.setattr(devices, "_TOOLTIP", None)
    assert devices.choices() == ["auto", "cuda:0", "xpu:0", "cpu"]
    assert devices.kinds() == ["cuda", "xpu"]
    assert devices.counts() == {"cuda": 1, "xpu": 1}
