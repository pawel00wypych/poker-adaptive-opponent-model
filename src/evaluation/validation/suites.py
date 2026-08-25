from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckSuite:
    mode: str
    baseline_only: bool = False
    requires_algorithm_coverage: bool = True
    requires_matchup_coverage: bool = True


SUITES = {
    "training-opponent": CheckSuite(mode="training-opponent"),
    "generalization": CheckSuite(mode="generalization"),
    "stress-test": CheckSuite(mode="stress-test"),
    "head-to-head": CheckSuite(mode="head-to-head"),
    "cross-play": CheckSuite(
        mode="cross-play",
        requires_matchup_coverage=False,
    ),
    "baseline-sanity": CheckSuite(
        mode="baseline-sanity",
        baseline_only=True,
        requires_algorithm_coverage=False,
        requires_matchup_coverage=False,
    ),
}
