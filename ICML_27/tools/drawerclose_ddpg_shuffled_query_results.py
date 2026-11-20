"""Publish completed DrawerClose DDPG Shuffled-Policy Query results against SABER."""

from drawerclose_ddpg_replay_query_results import SHUFFLED_QUERY, main


if __name__ == "__main__":
    main(SHUFFLED_QUERY)
