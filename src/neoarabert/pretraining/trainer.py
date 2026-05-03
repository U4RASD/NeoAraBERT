import os
import shutil
import re
from tqdm import tqdm

from omegaconf import OmegaConf, DictConfig, open_dict

# PyTorch
import torch
from torch.nn import CrossEntropyLoss

# Hugging Face
from datasets import load_from_disk
from transformers import BatchEncoding
from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from accelerate.utils import DistributedDataParallelKwargs

# Deepspeed
from deepspeed.utils import safe_get_full_fp32_param

# Our metric object and model
from .metrics import Metrics
from ..model import NeoAraBERTLMHead, NeoAraBERTConfig
from ..tokenizer import get_tokenizer
from ..optimizer import get_optimizer
from ..scheduler import get_scheduler
from ..dataloader import get_dataloader


def to_target_batch_size(
    batch: BatchEncoding,
    stored_batch: BatchEncoding,
    target_size: int = 8,
):
    tmp = {}
    batch_size = batch["input_ids"].shape[0]

    # If the batch is to large, we store samples
    if batch_size > target_size:
        for key in batch.keys():
            tmp[key] = torch.split(batch[key], [target_size, batch_size - target_size], dim=0)
            batch[key] = tmp[key][0]
            stored_batch[key] = tmp[key][1] if stored_batch[key] is None else torch.cat([tmp[key][1], stored_batch[key]], dim=0)

    # If the batch is to small, we fetch stored samples
    elif batch_size < target_size and stored_batch["input_ids"] is not None:
        stored_batch_size = stored_batch["input_ids"].shape[0]
        missing = target_size - batch_size

        # Fetch only necessary samples if storage is larger than required
        if missing < stored_batch_size:
            for key in batch.keys():
                stored_batch[key].to(batch[key].device)
                tmp[key] = torch.split(stored_batch[key], [missing, stored_batch_size - missing], dim=0)
                batch[key] = torch.cat([batch[key], tmp[key][0]], dim=0)
                stored_batch[key] = tmp[key][1]
                stored_batch[key].to("cpu", non_blocking=True)

        # Concatenate otherwise
        else:
            for key in batch.keys():
                batch[key] = torch.cat([batch[key], stored_batch[key]], dim=0)
                stored_batch[key] = None

    return batch, stored_batch


def _wandb_credentials_present() -> bool:
    """Return True iff a W&B API key is reachable without prompting.

    Checks the env var first, then ~/.netrc for an api.wandb.ai entry.
    Used to fail fast in trainer() when wandb.mode requires credentials
    but none are configured, instead of letting wandb prompt interactively
    and then crash mid-init.
    """
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        import netrc
        auth = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, OSError, Exception):
        return False
    return bool(auth and auth[2])


def _allow_unsafe_torch_load_for_accelerate_checkpoints():
    """Allow resuming Accelerate checkpoints under newer PyTorch 'weights_only' defaults.

    Newer PyTorch versions may default to `weights_only=True` (or Accelerate may opt into it),
    which prevents loading optimizer/scheduler states that contain non-tensor objects such as
    OmegaConf containers. Our checkpoints are created locally by this training code, so it's
    safe to load them with `weights_only=False` when resuming.
    """
    try:
        # 1) Allowlist OmegaConf container classes for PyTorch "weights_only" safe loading.
        # This avoids errors like:
        #   Unsupported global: GLOBAL omegaconf.listconfig.ListConfig
        try:
            from omegaconf import ListConfig, DictConfig as OmegaDictConfig

            try:
                torch.serialization.add_safe_globals([ListConfig, OmegaDictConfig])
            except Exception:
                pass
        except Exception:
            pass

        # Accelerate uses `accelerate.utils.other.load` internally during `accelerator.load_state()`.
        from accelerate.utils import other as acc_other
        import accelerate.checkpointing as acc_ckpt

        if getattr(acc_other, "_neoarabert_force_weights_only_false", False):
            return

        _orig_load = acc_other.load

        def _load(*args, **kwargs):
            # Force weights_only=False even if caller sets it.
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)

        acc_other.load = _load
        # Accelerate checkpointing may have imported `load` into its module scope.
        # Patch it too so load_state() uses our override.
        try:
            acc_ckpt.load = _load
        except Exception:
            pass
        acc_other._neoarabert_force_weights_only_false = True
    except Exception:
        # If anything goes wrong, don't block training; resume may fail and show the root error.
        return


def trainer(cfg: DictConfig):
    # Get the last checkpoint id
    checkpoint_dir = os.path.join(cfg.trainer.dir, "checkpoints")
    model_checkpoint_dir = os.path.join(cfg.trainer.dir, "model_checkpoints")
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    iteration = 0
    if cfg.trainer.resume and os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        # This regular expression was taken from accelerator.load_state()
        folders = os.listdir(checkpoint_dir)
        iteration = max(int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)[0]) for folder in folders) + 1
    elif int(os.environ.get("RANK", "0")) == 0 and os.path.exists(checkpoint_dir):
        # Fresh run: drop prior accelerate state so save_iteration=0 can write checkpoint_0
        # without colliding (accelerate also races between DDP ranks during rotate-and-save).
        shutil.rmtree(checkpoint_dir)

    # Accelerator object
    project_config = ProjectConfiguration(
        cfg.trainer.dir,
        automatic_checkpoint_naming=True,
        total_limit=cfg.trainer.accelerate.max_ckpt,
        iteration=iteration,
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # enable manual control of the scheduler
        mixed_precision=cfg.trainer.mixed_precision,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        log_with="wandb",
        project_config=project_config,
        kwargs_handlers=[kwargs],
    )

    os.makedirs(cfg.wandb.dir, exist_ok=True)
    api_key = cfg.wandb.get("api_key", None) if hasattr(cfg.wandb, "get") else getattr(cfg.wandb, "api_key", None)
    wandb_mode = str(cfg.wandb.mode) if cfg.wandb.mode is not None else "online"
    if api_key:
        import wandb as _wandb
        _wandb.login(key=str(api_key))
    elif wandb_mode in {"online", "shared", "auto"}:
        if not _wandb_credentials_present():
            raise RuntimeError(
                f"wandb.mode={wandb_mode!r} but no W&B API key was found. Either set "
                f"`wandb.api_key` in conf/neoarabert.yaml, export WANDB_API_KEY, run "
                f"`wandb login`, or set `wandb.mode=disabled` to skip wandb entirely."
            )
    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        init_kwargs={
            "wandb": {
                "name": cfg.wandb.name,
                "entity": cfg.wandb.entity,
                "config": OmegaConf.to_container(cfg) | {"distributed_type": accelerator.distributed_type},
                "tags": cfg.wandb.tags,
                "dir": cfg.wandb.dir,
                "mode": cfg.wandb.mode,
                "resume": cfg.wandb.resume,
            }
        },
    )

    # Set the seed
    set_seed(cfg.seed)

    # Enable TF32 on matmul and on cuDNN
    torch.backends.cuda.matmul.allow_tf32 = cfg.trainer.tf32
    torch.backends.cudnn.allow_tf32 = cfg.trainer.tf32

    # Local and global counters
    metrics = Metrics()
    accelerator.register_for_checkpointing(metrics)

    # Get the dtype for the pad_mask
    dtype_pad_mask = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype_pad_mask = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype_pad_mask = torch.bfloat16

    # Tokenizer
    tokenizer = get_tokenizer(**cfg.tokenizer)
    with open_dict(cfg.tokenizer):
        cfg.tokenizer.vocab_size = len(tokenizer)

    # Dataset
    train_dataset = load_from_disk(cfg.dataset.path_to_disk)

    # Dataloader
    train_dataloader = get_dataloader(train_dataset, tokenizer, dtype=dtype_pad_mask, **cfg.dataloader.train, **cfg.datacollator)
    
    # Keep handle to collate_fn for dynamic targeted masking
    collate_fn = getattr(train_dataloader, "collate_fn", None)
    
    # Configure static linear decay if enabled (sync total_steps from trainer config)
    if collate_fn is not None:
        mlm_collator = getattr(collate_fn, "mlm_collator", None)
        if mlm_collator is not None and hasattr(mlm_collator, "_mask_linear_decay_enabled"):
            if mlm_collator._mask_linear_decay_enabled:
                # Override total_steps with trainer's max_steps if not already set properly
                # We must account for gradient accumulation and number of workers because
                # the collator runs in worker processes and counts local batches (micro-steps).
                total_micro_batches = cfg.trainer.max_steps * cfg.trainer.gradient_accumulation_steps
                num_workers = cfg.dataloader.train.num_workers
                
                if num_workers > 0:
                    # Approximate batches per worker
                    collator_steps = total_micro_batches // num_workers
                else:
                    collator_steps = total_micro_batches

                mlm_collator.set_mask_linear_decay(True, collator_steps)
                if accelerator.is_main_process:
                    print(f"[Mask Linear Decay] Enabled with collator_steps={collator_steps} (max_steps={cfg.trainer.max_steps}, accum={cfg.trainer.gradient_accumulation_steps}, workers={num_workers})")

    # Model
    model = NeoAraBERTLMHead(NeoAraBERTConfig(**cfg.model, **cfg.tokenizer, pad_token_id=tokenizer.pad_token_id))
    accelerator.log({"model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})

    # Optimizer and Scheduler
    optimizer = get_optimizer(model, accelerator.distributed_type, name=cfg.optimizer.name, **cfg.optimizer.hparams)
    scheduler = get_scheduler(optimizer=optimizer, lr=cfg.optimizer.hparams.lr, **cfg.scheduler)

    # Prepare with accelerate
    train_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader,
        model,
        optimizer,
        scheduler,
    )

    # Loss function
    train_loss_fn = CrossEntropyLoss()

    # Resume from the latest checkpoint
    skipped_train_dataloader = None
    if cfg.trainer.resume and os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        _allow_unsafe_torch_load_for_accelerate_checkpoints()
        accelerator.load_state()
        train_dataloader.set_epoch(metrics["train/epochs"])
        skipped_train_dataloader = accelerator.skip_first_batches(train_dataloader, metrics["train/batches"] % len(train_dataloader))
        # If using static linear decay for targeted masking, seed the main-process
        # collator step counter from the resumed batch counter so wandb progress
        # is consistent immediately after resume. Worker-process collators recover
        # their own counters via the set_mask_linear_decay call below.
        if collate_fn is not None:
            mlm_collator = getattr(collate_fn, "mlm_collator", None)
            if mlm_collator is not None and getattr(mlm_collator, "_mask_linear_decay_enabled", False):
                num_workers = int(getattr(cfg.dataloader.train, "num_workers", 0) or 0)
                denom = max(1, num_workers)
                resume_collator_step = int(metrics["train/batches"]) // denom
                mlm_collator.set_mask_decay_step(resume_collator_step)

    # Progress bar
    pbar = tqdm(
        desc="Train",
        unit="step",
        initial=metrics["train/steps"],
        total=cfg.trainer.max_steps,
        disable=(cfg.trainer.disable_tqdm or not accelerator.is_main_process),
    )

    while cfg.trainer.max_steps > metrics["train/steps"]:
        # Use skipped_train_dataloader the first epoch after resuming
        dataloader = train_dataloader if skipped_train_dataloader is None else skipped_train_dataloader

        stored_batch = {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
        }
        i = 0
        for batch in dataloader:
            # Update number of batches
            metrics["train/batches"] += 1
            i += 1

            # Pack or truncate the batch to target batch size (batch size might be variable due to sequence packing).
            if batch["input_ids"].shape[0] != cfg.dataloader.train.batch_size:
                batch, stored_batch = to_target_batch_size(batch, stored_batch, cfg.dataloader.train.batch_size)

            # If it is still smaller, stored batches were not enough and we skip to the next iteration to fill the batch
            if batch["input_ids"].shape[0] < cfg.dataloader.train.batch_size:
                stored_batch = batch
                continue

            # Under the no_sync context manager, PyTorch will skip synchronizing the gradients when .backward() is
            # called, and the first call to .backward() outside this context manager will trigger the synchronization.
            # Accumulating manually gives more flexibility and is compatible with TPUs.
            if metrics["train/batches"] % cfg.trainer.gradient_accumulation_steps != 0:
                with accelerator.no_sync(model):
                    # Forward pass
                    logits = model(batch["input_ids"], batch.get("attention_mask", None))["logits"]
                    train_loss = train_loss_fn(logits.view(-1, cfg.tokenizer.vocab_size), batch["labels"].view(-1))

                    # Compute gradient
                    accelerator.backward(train_loss)

                    # Log metrics
                    metrics["train/local_samples"] += batch["input_ids"].shape[0]
                    if "attention_mask" in batch.keys():
                        metrics["train/local_tokens"] += (batch["attention_mask"] == 0).sum().item()
                    else:
                        metrics["train/local_tokens"] += batch["input_ids"].shape[1]
                    metrics["train/local_num_pred"] += (batch["labels"] != -100).sum().item()
                    metrics["train/local_sum_loss"] += train_loss.item() * (batch["labels"] != -100).sum().item()
                    metrics["train/local_num_correct"] += (logits.argmax(dim=-1) == batch["labels"]).sum().item()

            else:
                # Forward pass
                logits = model(batch["input_ids"], batch.get("attention_mask", None))["logits"]
                train_loss = train_loss_fn(logits.view(-1, cfg.tokenizer.vocab_size), batch["labels"].view(-1))

                # Compute gradient and apply clipping
                accelerator.backward(train_loss)
                if cfg.trainer.gradient_clipping is not None and cfg.trainer.gradient_clipping > 0:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clipping)

                # Log metrics
                pbar.update(1)
                metrics["train/steps"] += 1
                metrics["train/local_samples"] += batch["input_ids"].shape[0]
                if "attention_mask" in batch.keys():
                    metrics["train/local_tokens"] += (batch["attention_mask"] == 0).sum().item()
                else:
                    metrics["train/local_tokens"] += batch["input_ids"].shape[1]
                metrics["train/local_num_pred"] += (batch["labels"] != -100).sum().item()
                metrics["train/local_sum_loss"] += train_loss.item() * (batch["labels"] != -100).sum().item()
                metrics["train/local_num_correct"] += (logits.argmax(dim=-1) == batch["labels"]).sum().item()

                # Update the parameters and the scheduler
                optimizer.step()
                scheduler.step()
                
                # Sync static linear decay step for LOGGING purposes only.
                # The actual masking happens in worker processes with their own step counters,
                # but we need to update the main process collator's step so that
                # get_pos_metrics_for_logging() reports accurate progress to wandb.
                if collate_fn is not None:
                    mlm_collator = getattr(collate_fn, "mlm_collator", None)
                    if mlm_collator is not None and getattr(mlm_collator, "_mask_linear_decay_enabled", False):
                        # For logging, use global step progress (not micro-batch count)
                        # Scale to match the total_steps the collator was configured with
                        logging_step = int(metrics["train/steps"] * cfg.trainer.gradient_accumulation_steps)
                        if cfg.dataloader.train.num_workers > 0:
                            logging_step = logging_step // cfg.dataloader.train.num_workers
                        mlm_collator.set_mask_decay_step(logging_step)

                if metrics["train/steps"] % cfg.wandb.log_interval == 0:
                    if accelerator.distributed_type is DistributedType.DEEPSPEED:
                        metrics["train/grad_norm"] = model.get_global_grad_norm()
                        metrics["train/weight_norm"] = (
                            sum([safe_get_full_fp32_param(p).norm(2) ** 2 for p in model.parameters()]) ** 0.5
                        ).item()
                    # DDP
                    else:
                        metrics["train/grad_norm"] = (sum([p.grad.norm(2) ** 2 for p in model.parameters()]) ** 0.5).item()
                        metrics["train/weight_norm"] = (sum([p.norm(2) ** 2 for p in model.parameters()]) ** 0.5).item()

                    metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]

                    # Log per-POS masking metrics (static-mask progress + per-tag totals)
                    if collate_fn is not None:
                        mlm_collator = getattr(collate_fn, "mlm_collator", None)
                        if mlm_collator is not None and hasattr(mlm_collator, "get_pos_metrics_for_logging"):
                            for key, value in mlm_collator.get_pos_metrics_for_logging().items():
                                metrics[key] = value

                    metrics.log(accelerator)

                # Save the accelerator state from the main process
                if metrics["train/steps"] % cfg.trainer.accelerate.save_steps == 0:
                    accelerator.save_state()

                # Save the pytorch model
                if metrics["train/steps"] % cfg.trainer.model.save_steps == 0:
                    if accelerator.distributed_type is DistributedType.DEEPSPEED:
                        # DeepSpeed checkpointing is collective (ZeRO state sharded across ranks).
                        model.save_checkpoint(model_checkpoint_dir, tag=metrics["train/steps"])
                    else:
                        # DDP path: only the main process touches disk. Two ranks racing on the
                        # same directory caused multi-minute hangs on networked filesystems.
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            if cfg.trainer.model.max_ckpt is not None:
                                files = os.listdir(model_checkpoint_dir)
                                iterations = [int(f) for f in files if f.isdigit()]
                                iterations.sort()
                                while iterations is not None and len(iterations) >= cfg.trainer.model.max_ckpt:
                                    file_to_remove = iterations.pop(0)
                                    shutil.rmtree(os.path.join(model_checkpoint_dir, str(file_to_remove)))
                                    print(
                                        f"Deleted old model checkpoint {file_to_remove} due to limit " f"(max_ckpt = {cfg.trainer.model.max_ckpt})"
                                    )
                            path = os.path.join(model_checkpoint_dir, str(metrics["train/steps"]))
                            os.makedirs(path, exist_ok=True)
                            torch.save(
                                accelerator.unwrap_model(model).state_dict(),
                                os.path.join(path, "state_dict.pt"),
                            )
                        accelerator.wait_for_everyone()

                if metrics["train/steps"] >= cfg.trainer.max_steps:
                    break

                # Reset the gradient
                optimizer.zero_grad()

        # Log metrics
        metrics["train/epochs"] += 1

        # "Remove" the skipped dataloader once exhausted
        skipped_train_dataloader = None

    # Make sure that the wandb tracker finishes correctly and close the progress bar
    pbar.close()
    accelerator.end_training()
