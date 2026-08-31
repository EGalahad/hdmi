---
name: run-hdmi-experiments
description: Launch, monitor, resume, and diagnose HDMI training on remote GPU hosts. Use for HDMI PPO/PPO-ROA experiments, multi-node queues, W&B runs, remote asset/cache setup, experiment comparisons, and startup or runtime failures.
---

# Run HDMI experiments

Keep every HDMI run reproducible, remotely durable, and safe around existing workloads.

## Workflow

1. State the hypothesis, exact config, acceptance gate, and stop condition.
2. Check `nvidia-smi`, relevant processes, and `tmux` on every target host. Never overlap or restart a healthy run.
3. Record launch scripts, configs, commits, hosts, GPU IDs, W&B IDs, logs, plots, and decisions under `scripts/experiments/<date>_<name>/`.
4. Resolve Hugging Face assets once before DDP. Default to `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1`; use proxy port `7890` only when intentionally refreshing a missing cache.
5. Exclude live experiment `records/` and `outputs/` from any `rsync --delete` deployment.
6. Put long runs and sequential queues in named remote `tmux` sessions and pipe output through `tee` into the experiment records directory.
7. After launch, verify asset resolution, environment creation, W&B initialization, every expected rank, GPU utilization, and iteration progress. Process existence alone is insufficient.
8. On failure, preserve the log and evidence, stop only the exact failed session and children, confirm GPU release, fix the root cause, and relaunch with a fresh log and W&B ID.
9. Compare W&B curves only after enough iterations answer the decision question; retain generated plots and the verdict in the experiment log.

## Multi-node and monitoring rules

- Keep the queue/watch process on the remote master so later stages survive local disconnects.
- Allow only global rank 0 to initialize W&B and write shared checkpoints or metrics.
- Check full history for non-finite values and inspect checkpoint tensors at promotion gates.
- Do not weaken env count, module size, dataset, safety checks, or termination settings to rescue a release run unless the experiment explicitly tests that change.
- Treat checkpoint creation, stage transition, failure, and clean completion as monitoring gates; do not busy-poll healthy training.
