from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
from diffusers.optimization import get_scheduler
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from alphaanything.data.caption import RGBACaptionDataset, build_loader
from alphaanything.data.edit import RGBAEditDataset, build_edit_loader
from alphaanything.data.rgba import rgba_to_layerdiffuse_rgb
from alphaanything.models.flux2_klein import model_i2rgba, model_t2rgba
from alphaanything.models.flux2_klein.lora import configure_pipeline_for_lora
from alphaanything.models.flux2_klein.vae import SinglePassRGBAFlux2VAE
from alphaanything.modules.alpha_math import ea_score_to_alpha
from alphaanything.training.checkpoint import load_training_state, resolve_state_path, save_training_checkpoint
from alphaanything.training.losses import flow_ea_loss
from alphaanything.training.utils import (
    count_trainable_params,
    create_tensorboard_writer,
    gather_eval_records,
    gather_loss_logs,
    shard_eval_items,
    write_tensorboard_scalars,
)
from alphaanything.utils.cli import build_common_parser, resolve_cli_config, write_json, write_yaml
from alphaanything.utils.image import save_infer_sample, save_normalized_rgb, to01
from alphaanything.utils.model import cast_trainable_params_to_float32, resolve_weight_dtype

RGBADataset = RGBAEditDataset | RGBACaptionDataset


def _task_backend(task: str):
    if task == "image_to_rgba":
        return model_i2rgba
    if task == "text_to_rgba":
        return model_t2rgba
    raise ValueError(f"Unsupported task: {task!r}")


def sample_training_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler, batch_size: int, device: torch.device, flow_config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    timesteps = scheduler.timesteps.to(device=device)
    valid = (timesteps >= float(flow_config.get("min_timestep", 1))) & (timesteps <= float(flow_config.get("max_timestep", 999)))
    candidates = torch.nonzero(valid, as_tuple=False).flatten()
    indices = candidates[torch.randint(candidates.numel(), (batch_size,), device=device)]
    sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)[indices]
    return timesteps[indices], sigmas.view(batch_size, 1, 1, 1)


def compute_training_losses(
    *,
    batch: dict[str, Any],
    pipe: Flux2KleinPipeline,
    rgba_vae: torch.nn.Module,
    scheduler: FlowMatchEulerDiscreteScheduler,
    config: dict[str, Any],
    device: torch.device,
    weight_dtype: torch.dtype,
    backend: Any,
) -> dict[str, torch.Tensor]:
    target_rgba = batch[backend.TARGET_KEY].to(device=device, dtype=torch.float32, non_blocking=True)
    target_alpha = to01(target_rgba[:, 3:4])
    with torch.no_grad():
        raw_vae = rgba_vae.module if hasattr(rgba_vae, "module") else rgba_vae
        target_rgb = rgba_to_layerdiffuse_rgb(target_rgba)
        target = raw_vae.encode_normalized_mode(target_rgb.to(dtype=weight_dtype))
        reference = backend.encode_condition(raw_vae, batch, device=device, weight_dtype=weight_dtype) if backend.TASK == "image_to_rgba" else None
        embeds, text_ids, dropout_rate = backend.encode_text_features(pipe, batch["prompts"], config["model"], config["training"], device)

    timesteps, sigmas = sample_training_timesteps(scheduler, target.shape[0], device, config["flow"])
    noise = torch.randn_like(target)
    noisy = (1.0 - sigmas) * target.float() + sigmas * noise.float()
    if backend.TASK == "image_to_rgba":
        flow, _ = backend.forward(pipe, noisy.to(dtype=weight_dtype), reference, timesteps, embeds, text_ids)
    else:
        flow, _ = backend.forward(pipe, noisy.to(dtype=weight_dtype), timesteps, embeds, text_ids)

    flow_loss = (flow.float() - (noise.float() - target.float())).square().flatten(1).mean(1)
    predicted_x0 = noisy - sigmas * flow.float()
    if config["flow"].get("x0_clip") is not None:
        predicted_x0 = predicted_x0.clamp(-float(config["flow"]["x0_clip"]), float(config["flow"]["x0_clip"]))

    _, score = rgba_vae(
        predicted_x0,
        operation="decode_rgba",
        clamp_rgb=False,
        use_activation_checkpoint=bool(config["training"].get("vae_activation_checkpointing", False)),
    )
    losses = flow_ea_loss(flow_loss, score, target_alpha, sigmas, config["loss"])
    losses.update({"text_dropout_rate": dropout_rate, "sigma_mean": sigmas.mean(), "timestep_mean": timesteps.mean()})
    return losses


def _select_eval_indices(dataset: RGBADataset, evaluation: dict[str, Any], seed: int) -> list[int]:
    if evaluation.get("sample_indices") is not None:
        return [int(i) for i in evaluation["sample_indices"]]
    count = int(evaluation.get("inference_samples", 4))
    count = len(dataset) if count < 0 else min(count, len(dataset))
    if evaluation.get("selection", "first") == "random":
        return random.Random(seed).sample(range(len(dataset)), count)
    return list(range(count))


@torch.no_grad()
def run_eval(
    *, pipe: Flux2KleinPipeline, rgba_vae: SinglePassRGBAFlux2VAE, dataset: RGBADataset,
    config: dict[str, Any], accelerator: Accelerator, step: int, backend: Any,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    image_conditioned = backend.TASK == "image_to_rgba"
    directory = Path(config["checkpoint"]["output_dir"]) / "eval_infer" / f"step-{step}"
    seed = int(evaluation.get("inference_seed", config["runtime"].get("seed", 42)))
    records = []
    for number, index in shard_eval_items(_select_eval_indices(dataset, evaluation, seed), accelerator):
        example = dataset[index]
        target = example["target_pixel_values"] if image_conditioned else example["pixel_values"]
        condition = to_pil_image(to01(example["condition_pixel_values"])) if image_conditioned else None
        height, width = target.shape[-2:]
        with accelerator.autocast():
            args = (
                example["prompt"], height, width,
                int(evaluation.get("inference_steps", 50)), float(evaluation.get("guidance_scale", 4.0)),
                str(evaluation.get("negative_prompt", " ")),
                torch.Generator(device=accelerator.device).manual_seed(seed + number),
                accelerator.device, config["model"],
            )
            latent = backend.sample(pipe, condition, *args) if image_conditioned else backend.sample(pipe, *args)
            rgb, score = rgba_vae(latent, operation="decode_rgba")
        prefix = f"sample_{number:04d}"
        record = {
            "sample_number": number,
            "id": example.get("id", str(index)),
            "prompt": example["prompt"],
            "height": height,
            "width": width,
            "prediction_paths": save_infer_sample(directory, prefix, rgb[0], ea_score_to_alpha(score)[0]),
            "target_paths": save_infer_sample(directory, f"{prefix}_target", target[:3], to01(target[3:4])),
        }
        if image_conditioned:
            record["condition_path"] = save_normalized_rgb(directory / f"{prefix}_condition.png", example["condition_pixel_values"])
        records.append(record)

    records = gather_eval_records(records, accelerator)
    if not accelerator.is_main_process:
        return {}
    write_json(directory / "metadata.json", {"step": step, "seed": seed, "task": config["task"], "records": records})
    return {"eval_infer/num_samples": len(records), "step": step}


def train(config: dict[str, Any]) -> None:
    task = config.get("task", "image_to_rgba")
    backend = _task_backend(task)
    image_conditioned = task == "image_to_rgba"
    training = config["training"]
    model = config["model"]
    output_dir = Path(config["checkpoint"]["output_dir"])

    accelerator = Accelerator(
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        mixed_precision=training.get("mixed_precision", "bf16"),
        project_config=ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / "logs")),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    seed = int(config.get("runtime", {}).get("seed", 42))
    set_seed(seed, device_specific=True)

    dataset_cls = RGBAEditDataset if image_conditioned else RGBACaptionDataset
    loader_fn = build_edit_loader if image_conditioned else build_loader
    dataset = dataset_cls(config["data"])
    loader = loader_fn(
        dataset,
        {"train_batch_size": int(training.get("per_device_batch_size", 1)), "drop_last": True, "seed": seed},
        config["data"],
        shuffle=True,
    )
    if len(loader) < accelerator.num_processes:
        raise ValueError(f"Dataset yields {len(loader)} batches, fewer than {accelerator.num_processes} GPU processes.")

    source_key = "manifest_path" if image_conditioned else "dataset_name"
    eval_source = config["data"].get(f"eval_{source_key}")
    eval_dataset = (
        dataset_cls({**config["data"], source_key: eval_source or config["data"][source_key], "repeats": 1, "random_flip": False, "short_caption_probability": 0.0, "implicit_caption_probability": 0.0})
        if (eval_source or not image_conditioned)
        else dataset
    )

    weight_dtype = resolve_weight_dtype(training.get("mixed_precision", "bf16"))
    pipe = Flux2KleinPipeline.from_pretrained(
        model["pretrained_model_name_or_path"],
        torch_dtype=weight_dtype,
        local_files_only=bool(model.get("local_files_only", False)),
    )
    adapter_name = str(model["adapter_name"])
    configure_pipeline_for_lora(pipe, model, training, adapter_name)
    pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)

    rgba_vae = SinglePassRGBAFlux2VAE(pipe.vae, config["vae_alpha_adapter"]).to(accelerator.device)
    cast_trainable_params_to_float32(pipe.transformer)
    cast_trainable_params_to_float32(rgba_vae.alpha_decoder)

    flow_cfg = config.get("flow", {})
    flow_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=int(flow_cfg.get("num_train_timesteps", 1000)),
        shift=float(flow_cfg.get("shift", 3.0)),
        use_dynamic_shifting=False,
    )
    opt_cfg = config["optimizer"]
    base_lr = float(opt_cfg.get("learning_rate", 1e-4))
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in pipe.transformer.parameters() if p.requires_grad], "lr": float(opt_cfg.get("dit_learning_rate", base_lr))},
            {"params": [p for p in rgba_vae.parameters() if p.requires_grad], "lr": float(opt_cfg.get("vae_learning_rate", base_lr))},
        ],
        betas=(float(opt_cfg.get("beta1", 0.9)), float(opt_cfg.get("beta2", 0.999))),
        weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
        eps=float(opt_cfg.get("epsilon", 1e-8)),
    )
    max_steps = int(training.get("max_train_steps", 20000))
    lr_scheduler = get_scheduler(
        str(opt_cfg.get("lr_scheduler", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(opt_cfg.get("warmup_steps", 100)) * accelerator.num_processes,
        num_training_steps=max_steps * accelerator.num_processes,
    )

    resume = None
    if config["checkpoint"].get("resume_from"):
        resume = torch.load(resolve_state_path(config["checkpoint"]["resume_from"]), map_location="cpu", weights_only=True)
        load_training_state(resume, pipe.transformer, rgba_vae, adapter_name)

    pipe.transformer, rgba_vae, optimizer, loader, lr_scheduler = accelerator.prepare(
        pipe.transformer, rgba_vae, optimizer, loader, lr_scheduler,
    )
    step = 0
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        lr_scheduler.load_state_dict(resume["scheduler"])
        step = int(resume["step"])

    if accelerator.is_main_process:
        write_yaml(output_dir / "config.resolved.yaml", config)
        write_json(
            output_dir / "dataset_summary.json",
            {
                "task": task,
                "backbone": model["pretrained_model_name_or_path"],
                "num_samples": len(dataset),
                "bucket_counts": dataset.bucket_counts(),
                "trainable_transformer": count_trainable_params(accelerator.unwrap_model(pipe.transformer)),
                "trainable_alpha_branch": count_trainable_params(accelerator.unwrap_model(rgba_vae)),
            },
        )

    writer = create_tensorboard_writer(training.get("report_to", "tensorboard"), output_dir / "logs", accelerator)
    progress = tqdm(total=max_steps, initial=step, disable=not accelerator.is_local_main_process, desc=f"Flux2 Klein {task}")
    pipe.transformer.train()
    rgba_vae.train()

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            with accelerator.accumulate(pipe.transformer, rgba_vae):
                with accelerator.autocast():
                    losses = compute_training_losses(
                        batch=batch, pipe=pipe, rgba_vae=rgba_vae, scheduler=flow_scheduler,
                        config=config, device=accelerator.device, weight_dtype=weight_dtype, backend=backend,
                    )
                accelerator.backward(losses["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for group in optimizer.param_groups for p in group["params"]],
                        float(opt_cfg.get("max_grad_norm", 1.0)),
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            step += 1
            progress.update(1)

            if step % max(1, int(training.get("log_steps", 10))) == 0:
                logs = gather_loss_logs("train", losses, accelerator)
                logs.update({"step": step, "learning_rate/dit": optimizer.param_groups[0]["lr"], "learning_rate/vae": optimizer.param_groups[1]["lr"]})
                if accelerator.is_main_process:
                    accelerator.print(json.dumps(logs))
                    write_tensorboard_scalars(writer, logs, step)

            evaluation = config.get("evaluation", {})
            if evaluation.get("enabled", False) and step % max(1, int(evaluation.get("every_steps", 250))) == 0:
                accelerator.wait_for_everyone()
                wrapped = pipe.transformer
                pipe.transformer.eval()
                rgba_vae.eval()
                try:
                    pipe.transformer = accelerator.unwrap_model(wrapped)
                    logs = run_eval(
                        pipe=pipe, rgba_vae=accelerator.unwrap_model(rgba_vae), dataset=eval_dataset,
                        config=config, accelerator=accelerator, step=step, backend=backend,
                    )
                    if accelerator.is_main_process:
                        write_tensorboard_scalars(writer, logs, step)
                finally:
                    pipe.transformer = wrapped
                    pipe.transformer.train()
                    rgba_vae.train()
                accelerator.wait_for_everyone()

            save_steps = int(config["checkpoint"].get("save_steps", 1000))
            if save_steps > 0 and step % save_steps == 0:
                accelerator.wait_for_everyone()
                save_training_checkpoint(
                    accelerator=accelerator, transformer=pipe.transformer, rgba_vae=rgba_vae,
                    optimizer=optimizer, lr_scheduler=lr_scheduler, config=config, step=step,
                    destination=output_dir / f"checkpoint-{step}",
                )
                accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    save_training_checkpoint(
        accelerator=accelerator, transformer=pipe.transformer, rgba_vae=rgba_vae,
        optimizer=optimizer, lr_scheduler=lr_scheduler, config=config, step=step, destination=output_dir,
    )
    accelerator.wait_for_everyone()
    progress.close()
    if writer is not None:
        writer.close()
    accelerator.print(json.dumps({"status": "SUCCESS", "steps": step, "output_dir": str(output_dir)}))
    accelerator.end_training()


def main() -> None:
    parser = build_common_parser("Train FLUX.2 Klein RGBA LoRA and alpha branch.")
    args, unknown = parser.parse_known_args()
    train(resolve_cli_config(args, unknown))


if __name__ == "__main__":
    main()
