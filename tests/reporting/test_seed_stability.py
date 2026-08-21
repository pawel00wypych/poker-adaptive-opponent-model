import json

import pandas as pd
import pytest

from src.evaluation.metrics.seed_statistics import SEED_SPREAD_COLUMN
from src.evaluation.reporting.seed_stability import (
    RANKING_STABILITY_INSUFFICIENT,
    RANKING_STABILITY_MODERATE,
    SeedStabilityConfig,
    build_ranking_stability,
    build_seed_performance,
    build_seed_stability_report,
    calculate_kendalls_w,
    select_final_model_rows,
    write_seed_stability_outputs,
)
from tests.reporting.test_experiment_summary import make_game_row


def write_seed_stability_input(path):
    rows = []
    game_id = 0

    def add_agent_results(
        *,
        training_episode,
        agent,
        profits,
        seeds=(1, 2, 3),
        opponent="calling",
    ):
        nonlocal game_id
        for seed, profit in zip(seeds, profits, strict=True):
            rows.append(
                make_game_row(
                    seed=seed,
                    training_episode=training_episode,
                    agent=agent,
                    opponent=opponent,
                    game_id=game_id,
                    profit_bb=profit,
                )
            )
            game_id += 1

    add_agent_results(
        training_episode=2000,
        agent="adaptive_mc",
        profits=(10.0, 8.0, 6.0),
    )
    add_agent_results(
        training_episode=2000,
        agent="adaptive_q_learning",
        profits=(5.0, 7.0, 9.0),
    )
    add_agent_results(
        training_episode=2000,
        agent="adaptive_sarsa",
        profits=(0.0, 0.0, 0.0),
    )

    pd.DataFrame(rows).to_csv(path, index=False)


def make_seed_metric(
    *,
    seed,
    agent,
    mean_profit_bb,
):
    return {
        "training_run": "run",
        "model_source": "final",
        "training_episode": 2000,
        "opponent_name": "calling",
        "model_seed": seed,
        "agent_name": agent,
        "mean_profit_bb": mean_profit_bb,
    }


def test_calculate_kendalls_w_for_identical_and_reversed_rankings():
    identical = pd.DataFrame(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
    )
    reversed_rankings = pd.DataFrame(
        [
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
        ]
    )

    assert calculate_kendalls_w(identical) == pytest.approx(1.0)
    assert calculate_kendalls_w(reversed_rankings) == pytest.approx(0.0)
    assert calculate_kendalls_w(identical.iloc[:1]) is None


def test_calculate_kendalls_w_corrects_tied_rankings():
    identical_with_ties = pd.DataFrame(
        [
            [1.5, 1.5, 3.0],
            [1.5, 1.5, 3.0],
            [1.5, 1.5, 3.0],
        ]
    )
    rankings_without_any_order = pd.DataFrame(
        [
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
        ]
    )

    assert calculate_kendalls_w(identical_with_ties) == pytest.approx(1.0)
    assert calculate_kendalls_w(rankings_without_any_order) is None


def test_seed_report_uses_final_models_and_keeps_extreme_seed_ids(
    tmp_path,
):
    input_path = tmp_path / "results.csv"
    write_seed_stability_input(input_path)

    report, performance, summary, rankings, stability = build_seed_stability_report(
        input_path
    )

    assert report.overview["final_training_episodes"] == [2000]
    assert set(performance["training_episode"]) == {2000}
    assert set(rankings["training_episode"]) == {2000}

    monte_carlo = summary[summary["agent_name"] == "adaptive_mc"].iloc[0]
    assert monte_carlo["best_seed"] == 1
    assert monte_carlo["best_seed_mean_profit_bb"] == 10.0
    assert monte_carlo["worst_seed"] == 3
    assert monte_carlo["worst_seed_mean_profit_bb"] == 6.0
    assert monte_carlo[SEED_SPREAD_COLUMN] == 4.0
    assert monte_carlo["rank_seed_count"] == 3
    assert monte_carlo["first_place_count"] == 2
    assert monte_carlo["first_place_rate"] == pytest.approx(2 / 3)

    ranking_row = stability.iloc[0]
    assert ranking_row["complete_seed_count"] == 3
    assert ranking_row["excluded_seed_count"] == 0
    assert ranking_row["kendalls_w"] == pytest.approx(7 / 9)
    assert ranking_row["ranking_stability"] == RANKING_STABILITY_MODERATE


def test_final_model_rows_are_not_selected_by_episode(
    tmp_path,
):
    input_path = tmp_path / "results.csv"
    write_seed_stability_input(input_path)
    _, performance, _, _, _ = build_seed_stability_report(input_path)

    assert set(performance["training_episode"]) == {2000}

    selected = select_final_model_rows(performance)
    assert len(selected) == len(performance)
    assert set(selected["training_episode"]) == {2000}


def test_incomplete_seed_ranking_is_excluded_from_kendalls_w():
    metrics = pd.DataFrame(
        [
            make_seed_metric(seed=1, agent="agent_a", mean_profit_bb=3.0),
            make_seed_metric(seed=1, agent="agent_b", mean_profit_bb=2.0),
            make_seed_metric(seed=1, agent="agent_c", mean_profit_bb=1.0),
            make_seed_metric(seed=2, agent="agent_a", mean_profit_bb=3.0),
            make_seed_metric(seed=2, agent="agent_b", mean_profit_bb=2.0),
        ]
    )

    performance = build_seed_performance(metrics)
    stability = build_ranking_stability(performance)
    row = stability.iloc[0]

    assert performance.groupby("model_seed")["ranking_complete"].first().to_dict() == {
        1: True,
        2: False,
    }
    assert row["seed_count"] == 2
    assert row["complete_seed_count"] == 1
    assert row["excluded_seed_count"] == 1
    assert pd.isna(row["kendalls_w"])
    assert row["ranking_stability"] == RANKING_STABILITY_INSUFFICIENT


def test_report_keeps_rank_fields_when_no_seed_has_a_complete_ranking(
    tmp_path,
):
    input_path = tmp_path / "incomplete.csv"
    pd.DataFrame(
        [
            make_game_row(
                seed=1,
                training_episode=2000,
                agent="agent_a",
                opponent="calling",
                game_id=0,
                profit_bb=3.0,
            ),
            make_game_row(
                seed=2,
                training_episode=2000,
                agent="agent_b",
                opponent="calling",
                game_id=1,
                profit_bb=2.0,
            ),
        ]
    ).to_csv(input_path, index=False)

    report, _, summary, _, stability = build_seed_stability_report(input_path)

    assert "rank_seed_count" in summary.columns
    assert "mean_rank" in summary.columns
    assert set(summary["rank_seed_count"]) == {0}
    assert summary["mean_rank"].isna().all()
    assert report.seed_stability_summary[0]["mean_rank"] is None
    assert stability.iloc[0]["complete_seed_count"] == 0


def test_seed_stability_outputs_include_all_reports_and_valid_json(
    tmp_path,
):
    input_path = tmp_path / "results.csv"
    output_dir = tmp_path / "seed_stability"
    write_seed_stability_input(input_path)

    created = write_seed_stability_outputs(
        input_path=input_path,
        output_dir=output_dir,
        report_format="all",
        export_latex=True,
    )
    created_names = {path.name for path in created}

    assert created_names == {
        "seed_stability.md",
        "seed_stability.json",
        "seed_performance.csv",
        "seed_stability_summary.csv",
        "seed_rankings.csv",
        "ranking_stability.csv",
        "seed_stability_summary.tex",
        "seed_rankings.tex",
        "ranking_stability.tex",
    }

    json_text = (output_dir / "seed_stability.json").read_text(encoding="utf-8")
    payload = json.loads(json_text)
    assert "NaN" not in json_text
    assert payload["seed_stability_summary"][0]["best_seed"] is not None
    assert payload["ranking_stability"][0]["kendalls_w"] == pytest.approx(7 / 9)

    markdown = (output_dir / "seed_stability.md").read_text(encoding="utf-8")
    assert "# Seed stability report" in markdown
    assert "## Ranking stability" in markdown
    assert "Kendall's W" in markdown


def test_seed_stability_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="at least 2"):
        SeedStabilityConfig(min_complete_seeds_for_ranking=1)
