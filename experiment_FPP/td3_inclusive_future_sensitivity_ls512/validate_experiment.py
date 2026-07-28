from __future__ import annotations

import json
from pathlib import Path


FOLDER = Path(__file__).resolve().parent
SOURCE = FOLDER / "code" / "src"
REFERENCE = FOLDER.parent / "td3_saber_critic_only_v4_ls512" / "code" / "src"
REPLAY_REL = Path("metacog_lite_v0/replay.py")
EXCLUSIVE = "int(self.episode_step[s]) > step"
INCLUSIVE = "int(self.episode_step[s]) >= step"


def relative_files(root: Path) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def main() -> int:
    source_files = relative_files(SOURCE)
    reference_files = relative_files(REFERENCE)
    if source_files != reference_files:
        raise AssertionError(
            f"Source file set differs from validated TD3 SABER source: "
            f"only_here={sorted(source_files - reference_files)}, "
            f"only_reference={sorted(reference_files - source_files)}"
        )

    for relative in sorted(reference_files):
        source_path = SOURCE / relative
        reference_path = REFERENCE / relative
        if relative == REPLAY_REL:
            reference_text = reference_path.read_text()
            if reference_text.count(EXCLUSIVE) != 1:
                raise AssertionError("Expected exactly one exclusive future-goal predicate in reference replay.py")
            expected = reference_text.replace(EXCLUSIVE, INCLUSIVE)
            if source_path.read_text() != expected:
                raise AssertionError("Sensitivity replay.py differs by more than the inclusive predicate")
        elif source_path.read_bytes() != reference_path.read_bytes():
            raise AssertionError(f"Unexpected source difference: {relative}")

    config = json.loads((FOLDER / "config.json").read_text())
    assert config["future_goal_sampling"] == "inclusive_transition_next_achieved_goal"
    assert config["her_ratio"] == 0.8
    assert config["seeds"] == [1, 2, 3, 4, 5]
    assert config["total_steps"] == 1_000_000
    assert config["arms"]["her"] == {
        "trainer_method": "td3_her",
        "weight_scope": "none",
        "lambda_support": 0.0,
        "support_lookup": False,
        "run_slug": "td3_her_inclusive_future_ls512",
        "support_diagnostic_audit_interval": 100,
    }
    assert config["arms"]["saber"] == {
        "trainer_method": "td3_saber_runtime_opt",
        "weight_scope": "critic_only",
        "lambda_support": 1.0,
        "support_lookup": True,
        "run_slug": "td3_saber_inclusive_future_v4_ls512",
        "support_diagnostic_audit_interval": 0,
    }

    print("source parity and sensitivity configuration validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
