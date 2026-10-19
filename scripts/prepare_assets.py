"""Cache the selected task's assets before starting distributed workers."""
from importlib.util import find_spec
from pathlib import Path

import active_adaptation as aa
import hydra
from omegaconf import DictConfig, OmegaConf

MIMIC_CONFIG = Path(find_spec("mimic_lite").origin).parent.parent / "cfg"


@hydra.main(config_path=str(MIMIC_CONFIG), config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    if cfg.backend != "mjlab":
        raise ValueError("HDMI's training launcher currently supports MjLab")
    # Registering the projects resolves their robot asset references. This does
    # not construct an environment, initialize W&B, or allocate training state.
    aa.init(cfg, auto_rank=False)
    from any4hdmi.dataset.loading import resolve_input_paths

    motion_cfgs = OmegaConf.select(cfg, "task.command.motion_cfgs")
    if not motion_cfgs:
        raise ValueError("The selected task has no motion datasets")
    for name, motion in motion_cfgs.items():
        paths = resolve_input_paths(Path.cwd(), motion.path)
        if not paths or any(not path.exists() for path in paths):
            raise FileNotFoundError(f"Motion dataset {name!r} is unavailable: {paths}")
        print(f"Cached motion dataset: {name}", flush=True)


if __name__ == "__main__":
    main()
