"""Derive a local FK dataset with the v1.1 massless palm frames.

Public source motions and assets remain unchanged. Materialization is serialized
across DDP ranks and published atomically; the modified XML changes the FK key.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

from filelock import FileLock


def materialize_palm_dataset(source: Path, cache_dir: Path) -> Path:
    source = source.absolute()
    manifest = json.loads((source / "manifest.json").read_text())
    mjcf = manifest.get("mjcf")
    if not isinstance(mjcf, str) or Path(mjcf).is_absolute() or ".." in Path(mjcf).parts or mjcf.startswith("hf://"):
        raise ValueError("Palm derivation requires a dataset-relative MJCF")
    xml = (source / mjcf).read_bytes()
    tree = ET.fromstring(xml)
    for side in ("left", "right"):
        parent = tree.find(f".//body[@name='{side}_wrist_yaw_link']")
        if parent is None:
            raise ValueError(f"Missing {side}_wrist_yaw_link in {source}")
        palm = tree.find(f".//body[@name='{side}_palm_link']")
        if palm is None:
            ET.SubElement(parent, "body", name=f"{side}_palm_link", pos="0.1 0 0")
        elif palm not in list(parent) or palm.attrib.get("pos") != "0.1 0 0" or len(palm):
            raise ValueError(f"Existing {side} palm does not match the v1.1 fixed frame")
    # Source snapshots are immutable; local source changes also invalidate this view.
    digest = hashlib.sha256(b"hdmi-palm-v1\0" + str(source).encode() + xml)
    for path in sorted(source.rglob("*")):
        if path.is_file():
            stat = path.stat()
            digest.update(f"{path.relative_to(source)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / digest.hexdigest()
    with FileLock(str(target) + ".lock"):
        if not target.is_dir():
            temporary = Path(tempfile.mkdtemp(prefix="palm-", dir=cache_dir))
            try:
                # Dereference HF symlinks so the derived dataset is self-contained.
                shutil.copytree(source, temporary, dirs_exist_ok=True)
                ET.ElementTree(tree).write(temporary / mjcf, encoding="utf-8")
                (temporary / "palm_provenance.json").write_text(json.dumps({
                    "source": str(source), "source_mjcf_sha256": hashlib.sha256(xml).hexdigest(),
                    "offset_m": [0.1, 0, 0], "version": 1,
                }, indent=2))
                temporary.rename(target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    return target


def palm_motion_configs(configs):
    from any4hdmi.dataset.loading import resolve_input_paths
    import active_adaptation
    base_dir = Path(active_adaptation.__file__).parent.parent
    cache_dir = Path(os.environ.get("HDMI_DATASET_CACHE", Path.home() / ".cache/hdmi/palm-datasets"))
    result = {}
    for name, cfg in configs.items():
        cfg = dict(cfg)
        sources = resolve_input_paths(base_dir, cfg["path"])
        cfg["path"] = [str(materialize_palm_dataset(source, cache_dir)) for source in sources]
        result[name] = cfg
    return result
