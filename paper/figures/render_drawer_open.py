#!/usr/bin/env python3
"""Backward-compatible entry point for the DrawerOpen paper panel."""

import sys

from render_environment_panels import main


if __name__ == "__main__":
    raise SystemExit(main(["--task", "drawer-open", *sys.argv[1:]]))
