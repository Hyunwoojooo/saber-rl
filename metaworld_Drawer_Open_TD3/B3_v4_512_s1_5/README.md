# MetaWorld Drawer-Open-v3 TD3/B3 Code Snapshot

This folder contains the shared DrawerOpen training code used by the TD3
MetaWorld drawer-open experiments. It started as the FPP `B3_v4_512_s1_5`
port, but the actor-critic backbone is now TD3.

## Backbone

- Agent backend: TD3
- Twin critics with clipped double-Q target
- Target policy smoothing: `target_policy_noise = 0.2`, `target_noise_clip = 0.5`
- Delayed actor update: `policy_delay = 2`
- Historical imports still use `DDPGAgent`; this is an alias to `TD3Agent` in
  `code/src/metacog_lite_v0/agent.py`.

## DrawerOpen Task Adapter

MetaWorld `drawer-open-v3` is wrapped as a Fetch-style GoalEnv:

- `achieved_goal`: drawer handle position
- `desired_goal`: MetaWorld target position
- sparse training reward: `0` when handle-target distance is within threshold,
  otherwise `-1`
- evaluation logs both goal-derived success and MetaWorld environment success

The drawer-open support context is:

- coarse key: hand-handle distance bin, handle-y bin, gripper bin, horizon bin
- full context key: phase, hand-handle distance bin, handle-goal distance bin,
  handle-y bin, gripper bin, horizon bin

Phases: `approach`, `contact`, `opening`, `open`.

## Final Variants

Use the top-level runners from the repository root. Each script runs one
experiment with 5 parallel seeds:

```bash
bash metaworld_Drawer_Open_TD3/TD3-HER-B0-2M.sh
bash metaworld_Drawer_Open_TD3/TD3-HER-B3-Critic-Only-2M.sh
bash metaworld_Drawer_Open_TD3/TD3-HER-B3-Actor-Only-2M.sh
bash metaworld_Drawer_Open_TD3/TD3-HER-B3-Actor-Critic-2M.sh
bash metaworld_Drawer_Open_TD3/TD3-HER-B3-Random-Support-2M.sh
bash metaworld_Drawer_Open_TD3/TD3-HER-B3-Random-Fixed-Count-2M.sh
bash metaworld_Drawer_Open_TD3/TD3-HER-B3-Shuffled-Weight-2M.sh
```

## Required Dependency

MetaWorld must be installed in the Python environment used for the run. The
current runner prefers `.venv/bin/python` when it exists.
