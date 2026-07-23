# Environment renders

The figure directory contains three simulator-generated environment panels:

- `drawer_open_v3_render.png`: Meta-World DrawerOpen-v3
- `drawer_close_v3_render.png`: Meta-World DrawerClose-v3
- `fetch_pick_and_place_v4_render.png`: Gymnasium-Robotics
  FetchPickAndPlace-v4 (FPP)

The panels illustrate the experimental environments. They are not evaluation
frames and do not constitute performance evidence.

## Reproduce

From the repository root, run:

```bash
MUJOCO_GL=egl .venv/bin/python paper/figures/render_environment_panels.py
```

This command regenerates all three PNG files at `960 x 720`. Individual panels
can be generated with `--task drawer-open`, `--task drawer-close`, or
`--task fpp`. The legacy DrawerOpen command remains available:

```bash
MUJOCO_GL=egl .venv/bin/python paper/figures/render_drawer_open.py
```

The fixed presets are:

| Panel | Seed | Capture | Camera `(lookat; distance; azimuth; elevation)` |
|---|---:|---:|---|
| DrawerOpen-v3 | 1 | step 90 | `(-0.04, 0.68, 0.22); 1.50; 140; -23` |
| DrawerClose-v3 | 1 | step 82 | `(-0.04, 0.76, 0.22); 1.50; 140; -23` |
| FetchPickAndPlace-v4 | 3 | reset state | `(1.32, 0.75, 0.58); 1.70; 115; -22` |

The Drawer panels use Meta-World's bundled scripted policies only to place the
simulator in visually recognizable task states. FPP is rendered directly after
the seeded reset so that the robot, object, and target are visible. No learned
policy is used to generate any panel.

## LaTeX

```latex
\begin{figure*}[t]
  \centering
  \begin{minipage}{0.32\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/drawer_open_v3_render.png}
    \\[-1mm]\small (a) DrawerOpen
  \end{minipage}\hfill
  \begin{minipage}{0.32\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/drawer_close_v3_render.png}
    \\[-1mm]\small (b) DrawerClose
  \end{minipage}\hfill
  \begin{minipage}{0.32\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/fetch_pick_and_place_v4_render.png}
    \\[-1mm]\small (c) FetchPickAndPlace
  \end{minipage}
  \caption{Illustrative simulator renderings of the manipulation environments
  used in our experiments. These panels are not evaluation rollouts.}
  \label{fig:environments}
\end{figure*}
```
