"""Derive palm-augmented any4hdmi dataset roots.

Mimic-Lite 1.1 tracks ``left_palm_link`` / ``right_palm_link`` (massless child bodies
of the wrist yaw links in the current ``g1_xmls`` robot model). The public any4hdmi
OMOMO datasets were exported before those bodies existed, so their combined MJCF
(``g1-<object>.xml``) has no palm links and the any4hdmi FK cache built from it cannot
serve them. This module creates, next to the framework cache, a derived dataset root
that symlinks every file of the source root and replaces the MJCF with a copy that
carries the palm bodies. The any4hdmi FK cache is keyed on the MJCF content, so the
derived root gets its own cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

PALM_LINKS = {
    "left_wrist_yaw_link": ("left_palm_link", "0.1 0 0"),
    "right_wrist_yaw_link": ("right_palm_link", "0.1 0 0"),
}


def _framework_root() -> Path:
    import active_adaptation

    return Path(active_adaptation.__file__).resolve().parent.parent


def _derived_base_dir() -> Path:
    override = os.environ.get("HDMI_PALM_DATASET_ROOT")
    base = Path(override).expanduser() if override else _framework_root() / ".cache" / "palm_datasets"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _resolve_source_root(path: str | Path) -> Path:
    from any4hdmi.dataset.loading import find_any4hdmi_root, resolve_input_paths

    inputs = resolve_input_paths(_framework_root(), path)
    if len(inputs) != 1:
        raise ValueError(f"Expected one dataset root for {path!r}, got {len(inputs)}")
    root = find_any4hdmi_root(inputs[0])
    if root is None:
        raise FileNotFoundError(f"Missing any4hdmi manifest above {inputs[0]}")
    return Path(root).resolve()


def mjcf_has_palm_links(mjcf_path: Path) -> bool:
    tree = ET.parse(mjcf_path)
    names = {b.get("name") for b in tree.getroot().iter("body")}
    return all(palm in names for palm, _ in PALM_LINKS.values())


def add_palm_links(mjcf_path: Path, out_path: Path) -> None:
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    bodies = {b.get("name"): b for b in root.iter("body")}
    for wrist, (palm, pos) in PALM_LINKS.items():
        if palm in bodies:
            continue
        parent = bodies.get(wrist)
        if parent is None:
            raise ValueError(f"{mjcf_path} has no body {wrist!r}; cannot attach {palm!r}")
        ET.SubElement(parent, "body", name=palm, pos=pos)
    tree.write(out_path, encoding="unicode", xml_declaration=False)


def ensure_palm_dataset_root(path: str | Path) -> Path:
    """Return a dataset root whose MJCF has palm links (the source root if it already does)."""
    src = _resolve_source_root(path)
    manifest_path = src / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mjcf_ref = manifest.get("mjcf")
    if not isinstance(mjcf_ref, str) or mjcf_ref.startswith("hf://"):
        return src  # structured / remote MJCF references are left alone
    src_mjcf = (src / mjcf_ref).resolve()
    if mjcf_has_palm_links(src_mjcf):
        return src

    key = hashlib.sha256(str(src).encode()).hexdigest()[:12]
    name = str(manifest.get("dataset_name") or src.name)
    dst = _derived_base_dir() / f"{name}-palm-{key}"
    ready = dst / ".palm_ready"
    if ready.is_file():
        return dst

    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for entry in src.iterdir():
        if entry.name in (manifest_path.name, Path(mjcf_ref).name) or entry.name.startswith("."):
            continue
        (tmp / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    add_palm_links(src_mjcf, tmp / Path(mjcf_ref).name)
    shutil.copyfile(manifest_path, tmp / manifest_path.name)
    (tmp / ".palm_source").write_text(str(src) + "\n")
    (tmp / ".palm_ready").touch()
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    print(f"[hdmi][palm_datasets] derived {dst} from {src}")
    return dst


def palm_motion_cfgs(motion_cfgs: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite every ``path`` entry of a Mimic-Lite ``motion_cfgs`` mapping."""
    out: dict[str, Any] = {}
    for name, cfg in motion_cfgs.items():
        if not isinstance(cfg, Mapping) or "path" not in cfg:
            out[name] = cfg
            continue
        cfg = dict(cfg)
        p = cfg["path"]
        if isinstance(p, (str, Path)):
            cfg["path"] = str(ensure_palm_dataset_root(p))
        elif isinstance(p, (list, tuple)):
            cfg["path"] = [str(ensure_palm_dataset_root(q)) for q in p]
        out[name] = cfg
    return out


__all__ = ["ensure_palm_dataset_root", "palm_motion_cfgs", "add_palm_links", "mjcf_has_palm_links"]
