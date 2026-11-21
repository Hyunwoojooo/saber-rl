"""Publish completed DrawerClose DDPG Random-Support results against SABER."""

from drawerclose_ddpg_replay_query_results import RANDOM_SUPPORT, main


if __name__ == "__main__":
    main(RANDOM_SUPPORT)
