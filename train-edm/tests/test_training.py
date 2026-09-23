import json
import math
import sys
from dataclasses import replace
from fractions import Fraction

import pytest
import torch
from PIL import Image
from train_edm.__main__ import main
from train_edm.model import ModelConfig, UNet
from train_edm.training import (
    TrainConfig,
    _batch,
    _dataset,
    evaluate,
    load_model,
    train,
)


@pytest.fixture
def setup(tmp_path):
    model = ModelConfig(image_size=4, channels=1, width=4, multipliers=(1, 2))
    config = TrainConfig(batch_size=2, seed=7, ema_half_life=8, ema_rampup=1)
    data = tmp_path / "data.pt"
    images = torch.rand(5, 1, 4, 4, generator=torch.Generator().manual_seed(9)) * 2 - 1
    torch.save(images, data)
    return data, model, config


def test_resume_is_identical_to_uninterrupted_training(setup, tmp_path):
    data, model, config = setup
    global_state = torch.random.get_rng_state()
    full = train(data, tmp_path / "full", steps=4, model_config=model, config=config)
    split = train(data, tmp_path / "split", steps=2, model_config=model, config=config)
    train(data, split.parent, steps=4, resume=True, save_every=1)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    expected = torch.load(full, weights_only=True)
    actual = torch.load(split, weights_only=True)
    for field in ("model", "ema"):
        assert actual[field].keys() == expected[field].keys()
        for name in actual[field]:
            assert torch.equal(actual[field][name], expected[field][name]), (
                field,
                name,
            )
    for field in ("data_rng", "noise_rng"):
        assert torch.equal(actual[field], expected[field])
    for field in ("step", "train_config", "model_config", "data_signature"):
        assert actual[field] == expected[field]
    for parameter in expected["optimizer"]["state"]:
        for name, value in expected["optimizer"]["state"][parameter].items():
            assert torch.equal(actual["optimizer"]["state"][parameter][name], value)
    assert actual["optimizer"]["param_groups"] == expected["optimizer"]["param_groups"]
    assert not split.with_suffix(".tmp").exists()
    records = [
        json.loads(line)
        for line in (split.parent / "metrics.jsonl").read_text().splitlines()
    ]
    assert [record["step"] for record in records] == [1, 2, 3, 4]
    assert all(math.isfinite(record["loss"]) for record in records)


def test_training_ignores_ambient_dtype_and_preserves_rng(setup, tmp_path):
    data, model, config = setup
    original_dtype = torch.get_default_dtype()
    original_rng = torch.random.get_rng_state().clone()
    states = []
    try:
        for dtype in (torch.float32, torch.float64):
            torch.set_default_dtype(dtype)
            checkpoint = train(
                data,
                tmp_path / str(dtype),
                steps=2,
                model_config=model,
                config=config,
            )
            assert torch.get_default_dtype() == dtype
            assert torch.equal(torch.random.get_rng_state(), original_rng)
            states.append(torch.load(checkpoint, weights_only=True))
    finally:
        torch.set_default_dtype(original_dtype)
        torch.random.set_rng_state(original_rng)
    for field in ("model", "ema"):
        assert states[0][field].keys() == states[1][field].keys()
        for name, value in states[0][field].items():
            assert value.dtype == torch.float32
            assert states[1][field][name].dtype == torch.float32
            assert torch.equal(value, states[1][field][name]), (field, name)


def test_ema_matches_image_count_half_life(setup, tmp_path):
    data, model, config = setup
    checkpoint = train(
        data, tmp_path / "run", steps=1, model_config=model, config=config
    )
    first = torch.load(checkpoint, weights_only=True)
    train(data, checkpoint.parent, steps=2, resume=True)
    second = torch.load(checkpoint, weights_only=True)
    half_life = min(config.ema_half_life, 2 * config.batch_size * config.ema_rampup)
    beta = 0.5 ** (config.batch_size / half_life)
    for name in first["ema"]:
        expected = first["ema"][name].lerp(second["model"][name], 1 - beta)
        torch.testing.assert_close(second["ema"][name], expected, rtol=0, atol=0)
    assert any(
        not torch.equal(second["model"][name], second["ema"][name])
        for name in first["ema"]
    )


def test_load_and_validation_are_deterministic_and_preserve_state(setup, tmp_path):
    data, model, config = setup
    checkpoint = train(
        data, tmp_path / "run", steps=2, model_config=model, config=config
    )
    rng_state = torch.random.get_rng_state()
    for ema in (False, True):
        loaded = load_model(checkpoint, ema=ema)
        state = torch.load(checkpoint, weights_only=True)["ema" if ema else "model"]
        assert not loaded.training
        assert all(not parameter.requires_grad for parameter in loaded.parameters())
        for name, value in loaded.state_dict().items():
            assert torch.equal(value, state[name])
        loaded.train()
        loss = evaluate(loaded, data, batches=2, batch_size=2)
        assert loss == evaluate(loaded, data, batches=2, batch_size=2)
        assert loaded.training and math.isfinite(loss)
        assert all(parameter.grad is None for parameter in loaded.parameters())
    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_resume_rejects_changed_state_and_overwrite(setup, tmp_path):
    data, model, config = setup
    output = tmp_path / "run"
    checkpoint = train(data, output, steps=1, model_config=model, config=config)
    original = checkpoint.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        train(data, output, steps=2)
    with pytest.raises(ValueError, match="model configuration"):
        train(data, output, steps=2, resume=True, model_config=replace(model, width=8))
    with pytest.raises(ValueError, match="training configuration"):
        train(data, output, steps=2, resume=True, config=replace(config, lr=0.01))
    with pytest.raises(ValueError, match="saved step"):
        train(data, output, steps=1, resume=True)
    with pytest.raises(ValueError, match="same device type"):
        train(data, output, steps=2, resume=True, device="cuda")
    data.touch()
    with pytest.raises(ValueError, match="dataset changed"):
        train(data, output, steps=2, resume=True)
    assert checkpoint.read_bytes() == original


@pytest.mark.parametrize("channels", [1, 3])
def test_image_folder_transform_and_training(tmp_path, channels):
    folder = tmp_path / "images"
    folder.mkdir()
    Image.new("RGB", (9, 3), (255, 255, 255)).save(folder / "white.png")
    config = ModelConfig(image_size=4, channels=channels, width=4, multipliers=(1,))
    data, _ = _dataset(folder, config)
    batch = _batch(data, config, 2, torch.Generator().manual_seed(0))
    assert batch.shape == (2, channels, 4, 4) and batch.dtype == torch.float32
    assert (batch == 1).all()
    checkpoint = train(
        folder,
        tmp_path / "run",
        steps=1,
        model_config=config,
        config=TrainConfig(batch_size=2),
    )
    assert math.isfinite(
        evaluate(load_model(checkpoint), folder, batches=1, batch_size=1)
    )


@pytest.mark.parametrize(
    "bad",
    [
        torch.empty(0, 1, 4, 4),
        torch.zeros(2, 1, 4),
        torch.zeros(2, 3, 4, 4),
        torch.zeros(2, 1, 4, 4, dtype=torch.float64),
        torch.full((2, 1, 4, 4), 2.0),
        torch.full((2, 1, 4, 4), float("nan")),
        {"not": "a tensor"},
    ],
)
def test_rejects_invalid_data(setup, tmp_path, bad):
    data, model, config = setup
    torch.save(bad, data)
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        train(data, output, steps=1, model_config=model, config=config)
    assert not output.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 0),
        ("batch_size", True),
        ("seed", -1),
        ("seed", 2**64 - 1),
        ("seed", 2**64),
        ("lr", 0),
        ("lr", float("inf")),
        ("ema_half_life", -1),
        ("ema_rampup", 0),
        ("p_mean", float("nan")),
        ("p_std", -1),
    ],
)
def test_rejects_invalid_config(field, value):
    with pytest.raises((ValueError, TypeError)):
        TrainConfig(**{field: value})


def test_checkpoint_metadata_uses_python_scalars(setup, tmp_path):
    data, model, config = setup
    model = replace(model, sigma_data=Fraction(1, 2))
    config = replace(config, lr=Fraction(1, 5000), p_mean=Fraction(-6, 5))
    checkpoint = train(
        data, tmp_path / "run", steps=1, model_config=model, config=config
    )
    state = torch.load(checkpoint, weights_only=True)
    assert type(state["model_config"]["sigma_data"]) is float
    assert type(state["train_config"]["lr"]) is float
    assert type(state["train_config"]["p_mean"]) is float
    assert load_model(checkpoint).config == model
    train(data, checkpoint.parent, steps=2, resume=True)


def test_evaluate_rejects_unrepresentable_seed(setup):
    data, model, _ = setup
    with pytest.raises(ValueError, match="seed must be less"):
        evaluate(UNet(model), data, seed=2**64 - 1)


def test_rejects_unsupported_checkpoint(tmp_path):
    checkpoint = tmp_path / "old.pt"
    torch.save({"format": 0}, checkpoint)
    with pytest.raises(ValueError, match="unsupported"):
        load_model(checkpoint)


def test_cli_train_resume_evaluate_and_sample(setup, tmp_path, monkeypatch):
    data, _, _ = setup
    output = tmp_path / "cli"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train-edm",
            "train",
            "--data",
            str(data),
            "--output",
            str(output),
            "--steps",
            "1",
            "--size",
            "4",
            "--channels",
            "1",
            "--width",
            "4",
            "--multipliers",
            "1",
            "2",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ],
    )
    main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train-edm",
            "resume",
            "--data",
            str(data),
            "--output",
            str(output),
            "--steps",
            "2",
            "--device",
            "cpu",
        ],
    )
    main()
    checkpoint = output / "checkpoint.pt"
    assert torch.load(checkpoint, weights_only=True)["step"] == 2
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train-edm",
            "evaluate",
            "--data",
            str(data),
            "--checkpoint",
            str(checkpoint),
            "--batches",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ],
    )
    main()
    samples = tmp_path / "samples"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train-edm",
            "sample",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(samples),
            "--count",
            "2",
            "--steps",
            "3",
            "--sigma-max",
            "1",
            "--device",
            "cpu",
        ],
    )
    main()
    generated = torch.load(samples / "samples.pt", weights_only=True)
    assert generated.shape == (2, 1, 4, 4) and torch.isfinite(generated).all()
    with Image.open(samples / "000000.png") as preview:
        assert preview.mode == "L" and preview.size == (4, 4)
    with pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_available_accelerator_training_resume_and_inference(setup, tmp_path, device):
    available = (
        torch.backends.mps.is_available()
        if device == "mps"
        else torch.cuda.is_available()
    )
    if not available:
        pytest.skip(f"{device} unavailable")
    data, model, config = setup
    checkpoint = train(
        data,
        tmp_path / device,
        steps=1,
        model_config=model,
        config=config,
        device=device,
    )
    train(data, checkpoint.parent, steps=2, resume=True, device=device)
    loaded = load_model(checkpoint, device=device)
    assert next(loaded.parameters()).device.type == device
    assert math.isfinite(evaluate(loaded, data, batches=1, batch_size=2))
