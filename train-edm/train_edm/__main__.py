"""CLI: python -m train_edm {train,resume,sample,evaluate}."""

import argparse
import json
from pathlib import Path

import torch

from train_edm.model import ModelConfig
from train_edm.sampling import sample_edm
from train_edm.training import TrainConfig, evaluate, load_model, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train a new unconditional denoiser")
    resume = commands.add_parser("resume", help="Resume optimizer, EMA, and RNG state")
    sampling = commands.add_parser(
        "sample", help="Generate raw tensors and PNG previews"
    )
    validation = commands.add_parser(
        "evaluate", help="Estimate held-out EDM loss, not FID"
    )
    for command in (training, resume, sampling, validation):
        command.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )
    for command in (training, resume, validation):
        command.add_argument("--data", type=Path, required=True)
    for command in (training, resume):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument(
            "--steps", type=int, required=True, help="Total optimizer steps"
        )
        command.add_argument("--save-every", type=int, default=1000)
    for command in (training, sampling, validation):
        command.add_argument("--seed", type=int, default=0)
    for command in (training, validation):
        command.add_argument("--batch-size", type=int, default=64)
        command.add_argument("--p-mean", type=float, default=-1.2)
        command.add_argument("--p-std", type=float, default=1.2)
    training.add_argument("--size", type=int, default=32)
    training.add_argument("--channels", type=int, default=3)
    training.add_argument("--width", type=int, default=32)
    training.add_argument("--multipliers", type=int, nargs="+", default=(1, 2, 2))
    training.add_argument("--sigma-data", type=float, default=0.5)
    training.add_argument("--lr", type=float, default=2e-4)
    training.add_argument("--ema-half-life", type=float, default=500_000)
    training.add_argument("--ema-rampup", type=float, default=0.05)
    for command in (sampling, validation):
        command.add_argument("--checkpoint", type=Path, required=True)
    sampling.add_argument("--output", type=Path, required=True)
    sampling.add_argument("--count", type=int, default=16)
    sampling.add_argument("--steps", type=int, default=18)
    sampling.add_argument("--sigma-min", type=float, default=0.002)
    sampling.add_argument("--sigma-max", type=float, default=80.0)
    sampling.add_argument("--rho", type=float, default=7.0)
    validation.add_argument("--batches", type=int, default=16)
    args = parser.parse_args()
    if args.command in ("train", "resume"):
        model_config = config = None
        if args.command == "train":
            model_config = ModelConfig(
                image_size=args.size,
                channels=args.channels,
                width=args.width,
                multipliers=tuple(args.multipliers),
                sigma_data=args.sigma_data,
            )
            config = TrainConfig(
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                p_mean=args.p_mean,
                p_std=args.p_std,
                ema_half_life=args.ema_half_life,
                ema_rampup=args.ema_rampup,
            )
        train(
            args.data,
            args.output,
            steps=args.steps,
            model_config=model_config,
            config=config,
            device=args.device,
            resume=args.command == "resume",
            save_every=args.save_every,
        )
        return
    model = load_model(args.checkpoint, device=args.device)
    if args.command == "evaluate":
        loss = evaluate(
            model,
            args.data,
            batches=args.batches,
            batch_size=args.batch_size,
            seed=args.seed,
            p_mean=args.p_mean,
            p_std=args.p_std,
        )
        print(json.dumps({"edm_loss": loss, "batches": args.batches}))
        return
    if args.output.exists():
        parser.error("sample output already exists; choose a new directory")
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(
        args.seed
    )
    images = sample_edm(
        model,
        args.count,
        generator=generator,
        num_steps=args.steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
    ).cpu()
    args.output.mkdir(parents=True)
    torch.save(images, args.output / "samples.pt")
    if model.config.channels in (1, 3):
        from PIL import Image

        for index, image in enumerate(images):
            pixels = ((image + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
            pixels = bytes(pixels.permute(1, 2, 0).contiguous().flatten().tolist())
            preview = Image.frombytes(
                "L" if model.config.channels == 1 else "RGB",
                (model.config.image_size,) * 2,
                pixels,
            )
            preview.save(args.output / f"{index:06d}.png")
    print(json.dumps({"output": str(args.output), "count": len(images)}))


if __name__ == "__main__":
    main()
