"""Finding and fetching Whisper large-v3 turbo, without its 1.6 GB of weights."""

from __future__ import annotations

import os

import pytest

from yue2_comfy import constants, discovery, download, paths

SIZE = 64
"""The weights' size in these tests: the release's is stood in for, since the folder is known by it alone."""


@pytest.fixture(autouse=True)
def small_release(monkeypatch):
    monkeypatch.setattr(discovery, "WHISPER_BYTES", SIZE)


def whisper_folder(folder, size=SIZE, generation=True):
    """A folder that looks like the release: weights of the right size, the tokenizer and the generation config."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / constants.WEIGHTS_NAME).write_bytes(b"\0" * size)
    (folder / constants.WHISPER_TOKENIZER_NAME).write_text("{}", encoding="utf-8")
    if generation:
        (folder / constants.WHISPER_GENERATION_NAME).write_text("{}", encoding="utf-8")
    return folder


def test_the_downloaded_folder_is_found_under_the_models_root(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    folder = whisper_folder(tmp_path / constants.WHISPER_DIRNAME)
    assert discovery.find_whisper() == str(folder)


def test_a_fine_tune_of_another_size_is_not_taken_for_it(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    whisper_folder(tmp_path / "whisper-large-v3-turbo-russian", size=SIZE * 2)
    assert discovery.find_whisper() == ""


def test_weights_without_their_generation_config_are_not_enough(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    whisper_folder(tmp_path / constants.WHISPER_DIRNAME, generation=False)
    assert discovery.find_whisper() == ""


def test_a_collection_beside_the_checkout_is_searched(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.ENV_ROOT, raising=False)
    monkeypatch.setattr(paths, "_folder_paths", lambda: None)
    monkeypatch.setattr(paths, "hf_cache_roots", lambda: [])
    monkeypatch.setattr(paths, "checkout_sibling_root", lambda: str(tmp_path))
    whisper_folder(tmp_path / "whisper" / "whisper-large-v3", size=SIZE * 2)
    folder = whisper_folder(tmp_path / "whisper" / "whisper-large-v3-turbo")
    assert discovery.find_whisper() == str(folder)


def test_the_refusal_names_the_links_the_folder_and_where_it_looked(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    message = discovery.whisper_missing_message([str(tmp_path)])
    for name in constants.WHISPER_FILES:
        assert "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/" + name in message
    assert os.path.join(str(tmp_path), constants.WHISPER_DIRNAME) in message
    assert "Looked in 1 place" in message and "MIT" in message


def test_the_download_lands_in_its_own_folder_under_the_models_root(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "models_root", lambda: str(tmp_path))
    fetched = []
    monkeypatch.setattr(download, "fetch",
                        lambda repo, wanted, title, progress=None: fetched.append((repo, wanted, title)))
    assert download.fetch_whisper() == str(tmp_path / constants.WHISPER_DIRNAME)
    repo, wanted, title = fetched[0]
    assert repo == "openai/whisper-large-v3-turbo"
    assert wanted == {name: str(tmp_path / constants.WHISPER_DIRNAME / name) for name in constants.WHISPER_FILES}
    assert "Whisper" in title and "GB" in title


def test_nothing_is_fetched_when_the_folder_is_here(monkeypatch):
    monkeypatch.setattr(discovery, "find_whisper", lambda roots=None: "here")
    monkeypatch.setattr(download, "fetch_whisper", lambda progress=None: pytest.fail("fetched"))
    assert download.ensure_whisper({"download": "off"}) == "here"


def test_downloading_off_explains_instead_of_fetching(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    monkeypatch.setattr(download, "fetch_whisper", lambda progress=None: pytest.fail("fetched"))
    with pytest.raises(FileNotFoundError) as refused:
        download.ensure_whisper({"download": "off"})
    assert constants.WHISPER_DIRNAME in str(refused.value) and "'download' in YuE2 Options" in str(refused.value)


@pytest.mark.parametrize("choice", ["auto", "comfy-org", "original"])
def test_every_other_download_choice_fetches_openais_release(monkeypatch, tmp_path, choice):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    monkeypatch.setattr(download, "fetch_whisper", lambda progress=None: "fetched")
    assert download.ensure_whisper({"download": choice}) == "fetched"
