"""Where the sampler's generator is made and what it waits on, on each device.

``torch.multinomial`` refuses a generator from another device, so a request's
generator has to be made on the device its probabilities live on; on an Intel
card that is the card itself, and making it on the processor -- which is what
the sampler did before XPU was named -- is a refusal at the first token. The
device arithmetic is checked here; that an XPU generator really is accepted by
XPU multinomial is a property of the card.
"""

from __future__ import annotations

import pytest

sampling = pytest.importorskip("yue2_comfy.vendor.yue2.sampling")
torch = pytest.importorskip("torch")


def known(name):
    """Whether this torch build knows the device type at all."""
    try:
        torch.device(name)
    except Exception:
        return False
    return True


def test_the_generator_is_made_on_the_device_the_probabilities_live_on():
    assert sampling.rng_device("cpu") == torch.device("cpu")
    assert sampling.rng_device(torch.device("cpu")) == torch.device("cpu")
    if known("cuda"):
        assert sampling.rng_device("cuda:1") == torch.device("cuda:1")
    if known("xpu"):
        assert sampling.rng_device("xpu:1") == torch.device("xpu:1")


def test_a_device_without_a_generator_samples_through_the_processor():
    if not known("mps"):
        pytest.skip("this torch has no MPS device type")
    assert sampling.rng_device("mps") == torch.device("cpu")


def test_synchronize_waits_on_the_backend_the_device_names(monkeypatch):
    if not known("xpu"):
        pytest.skip("this torch has no XPU device type")
    waited = []
    monkeypatch.setattr(torch.xpu, "synchronize", lambda device=None: waited.append(device))
    sampling.synchronize("xpu:0")
    assert waited and waited[0].type == "xpu"


def test_synchronize_says_nothing_on_the_processor():
    assert sampling.synchronize("cpu") is None
