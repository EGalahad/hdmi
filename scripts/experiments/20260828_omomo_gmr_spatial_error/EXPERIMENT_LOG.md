# OMOMO GMR spatial-error training

- Status: dataset-mesh runtime accepted; queued for a fully idle 8-GPU host.
- Task: `hdmi-base`, `object=suitcase`, accepted 333-motion OMOMO GMR view.
- Scale: 8 GPUs x 8192 envs, seed 0, PPO large, 1000 iterations.
- Checkpoints: 250, 500, 750, and 1000.
- Source: AA `31b151d2`, Mimic-Lite `cdcef4d`, HDMI `f649f02`.
- Dataset manifest SHA-256: `ac63d7eaef6744200f4236de583cfd8f46f9553139c3160e20da005f0806aeff`.
- Observation: policy 544, command 376, priv 580; actor input 920, critic input 1500.
- Runtime object: separate MjLab entity using the accepted dataset mesh/frame.
- Host/tmux/W&B: pending launch verification.
