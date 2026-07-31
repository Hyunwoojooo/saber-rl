from __future__ import annotations

import unittest

from _source_import import add_fpp_ddpg

add_fpp_ddpg()
import test_b3_runtime_opt_lookup  # noqa: E402


class SupportMembershipReferenceTest(unittest.TestCase):
    def test_optimized_lookup_matches_exact_reference(self) -> None:
        self.assertEqual(test_b3_runtime_opt_lookup.main(), 0)


if __name__ == "__main__":
    unittest.main()
