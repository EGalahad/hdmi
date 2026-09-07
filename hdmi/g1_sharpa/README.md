# Sharpa Wave model assets

This directory vendors the MuJoCo Menagerie description of the left and right
Sharpa Wave hands for the G1 + Sharpa MJLab environment.

## Provenance

- Runtime model source: `google-deepmind/mujoco_menagerie/sharpa_wave`
- Runtime source commit: `8161bba264d7fa7c99ca301e91e7fb44737676ad`
- Retrieved: 2026-09-06
- Manufacturer source: `sharpa-robotics/sharpa-urdf-usd-xml`
- Manufacturer source commit used by Menagerie:
  `6eea427eb24189519f32b9f21674cd534d3f973c`
- License: Apache License 2.0; see `LICENSE`.

The unmodified upstream model notes and changelog are retained as
`xmls/UPSTREAM_README.md` and `xmls/UPSTREAM_CHANGELOG.md`. Menagerie's
`make_right.py` is retained beside the XML files because it defines and verifies
the right-hand mirror transformation.

## Included

- Menagerie `left_hand.xml` and generated `right_hand.xml`.
- Left- and right-hand visual and collision meshes.
- Twenty-two position actuators per hand.
- Fitted finger collision capsules and a VHACD palm collision decomposition.
- Upstream intra-hand contact exclusions and MuJoCo dynamics parameters.

## Local integration

The upstream hand files are kept unchanged. G1 wrist attachment, `lh/` and
`rh/` namespaces, and the fixed wrist-to-palm contact exclusions are applied at
runtime by `spec.py`. The composed model also explicitly adopts Menagerie's
`implicitfast` integrator, elliptic friction cone, and `impratio=10`; MuJoCo
otherwise retains the G1 parent's global options when attaching a child spec.
Active Adaptation actuator groups and canonical simulation ordering are defined
in `asset.py`.

## Standalone control task

Phase 8 adds the MJLab task `g1-sharpa-control`. Its 73-dimensional joint
position action is deliberately split into stable body, left-hand, and
right-hand groups of 29, 22, and 22 joints. Run a scripted headless check with:

```bash
uv run --project venv/mjlab python projects/hdmi/scripts/run_g1_sharpa_control.py \
  --num-envs 64 --duration 0.5 --mode sequence --device cuda
```

Add `--viewer --realtime` and use a longer duration for visual inspection. The
available motion modes are `hold`, `body`, `left`, `right`, `both`, and
`sequence`.

## HDMI integration

The Phase 9 task `omomo-suitcase-object-pose-sharpa` inherits the existing
OMOMO suitcase task. The HDMI policy retains its original 29-dimensional G1
body action and body-only observations; a separate command-prescribed
44-dimensional input holds both Sharpa hands at their default joint pose. This
lets the body policy be retrained for the hands' mass, inertia, and contacts
without requiring finger reference motion.

Future G1 connector geometry should be maintained separately and may use the
manufacturer's `with_flange` or `with_wrist` assets as its mechanical reference.
