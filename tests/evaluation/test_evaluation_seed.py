import pytest

from src.evaluation.runners.checkpoint_evaluator import build_game_seed
from src.evaluation.runners.cross_play_evaluator import (
    build_cross_play_seed,
)
from src.evaluation.runners.evaluation_seed import (
    build_paired_evaluation_seed,
)
from src.evaluation.runners.generalization_evaluator import (
    build_generalization_seed,
)
from src.evaluation.runners.head_to_head_evaluator import (
    build_head_to_head_seed,
)
from src.evaluation.runners.stress_test_evaluator import (
    build_stress_test_seed,
)


def test_build_paired_evaluation_seed_is_deterministic():
    first = build_paired_evaluation_seed(
        eval_seed_base=100_000,
        model_seed=42,
        checkpoint_episode=5000,
        matchup_game_index=7,
    )
    second = build_paired_evaluation_seed(
        eval_seed_base=100_000,
        model_seed=42,
        checkpoint_episode=5000,
        matchup_game_index=7,
    )

    assert first == second
    assert 0 <= first <= 2**32 - 1


@pytest.mark.parametrize(
    "runner_seed_builder",
    [
        build_game_seed,
        build_generalization_seed,
        build_head_to_head_seed,
        build_stress_test_seed,
        build_cross_play_seed,
    ],
)
def test_all_runner_seed_builders_use_the_shared_pairing_scheme(
    runner_seed_builder,
):
    arguments = {
        "eval_seed_base": 100_000,
        "model_seed": 42,
        "checkpoint_episode": 5000,
        "matchup_game_index": 7,
    }

    assert runner_seed_builder(**arguments) == (
        build_paired_evaluation_seed(**arguments)
    )


@pytest.mark.parametrize(
    ("component", "value"),
    [
        ("eval_seed_base", 100_001),
        ("model_seed", 43),
        ("checkpoint_episode", 5001),
        ("matchup_game_index", 8),
    ],
)
def test_build_paired_evaluation_seed_uses_every_component(
    component,
    value,
):
    arguments = {
        "eval_seed_base": 100_000,
        "model_seed": 42,
        "checkpoint_episode": 5000,
        "matchup_game_index": 7,
    }
    reference = build_paired_evaluation_seed(**arguments)
    arguments[component] = value

    assert build_paired_evaluation_seed(**arguments) != reference


@pytest.mark.parametrize(
    "invalid_value",
    [-1, 1.5, True],
)
def test_build_paired_evaluation_seed_rejects_invalid_components(
    invalid_value,
):
    expected_exception = (
        ValueError if invalid_value == -1 else TypeError
    )

    with pytest.raises(expected_exception):
        build_paired_evaluation_seed(
            eval_seed_base=100_000,
            model_seed=42,
            checkpoint_episode=5000,
            matchup_game_index=invalid_value,
        )
