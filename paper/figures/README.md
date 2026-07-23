# DrawerOpen environment render

`drawer_open_v3_render.png` is an illustrative rendering of the exact
Meta-World task family used by the DrawerOpen experiments. It is not an
evaluation frame and does not constitute performance evidence.

## Reproduce

From the repository root, run:

```bash
MUJOCO_GL=egl .venv/bin/python paper/figures/render_drawer_open.py
```

The renderer uses:

- environment: `Meta-World/MT1`
- task: `drawer-open-v3`
- reward function version: `v2`
- seed: `1`
- camera: `corner4`
- resolution: `960 x 720`
- capture step: `90`
- controller: Meta-World's bundled `SawyerDrawerOpenV3Policy`

The scripted controller is used only to place the simulator in a visually
recognizable opened-drawer state. The paper should describe the panel as an
illustrative environment rendering rather than a learned-policy rollout.

## LaTeX

```latex
\includegraphics[width=\linewidth]{figures/drawer_open_v3_render.png}
```

Suggested caption:

> Illustrative rendering of the Meta-World DrawerOpen-v3 environment used in
> our experiments. The image is generated from the simulator and is not an
> evaluation result.
