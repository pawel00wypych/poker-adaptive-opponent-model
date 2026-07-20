import pandas as pd
import pytest

from src.evaluation.result_logger import ResultLogger


def test_result_logger_writes_extended_game_data(
    tmp_path,
):
    output_path = tmp_path / "results.csv"

    logger = ResultLogger(
        str(output_path)
    )

    logger.log_game(
        experiment_name="adaptive_mc_vs_calling",
        game_id=1,
        agent_name="adaptive_mc",
        opponent_name="calling",
        final_stack=0,
        initial_stack=200,
        hands_played=25,
        big_blind=10,
        ended_by_bust=True,
        ended_by_round_limit=False,
        classified_decisions=8,
        correct_classifications=6,
        incorrect_classifications=2,
        unknown_classifications=3,
        classifier_accuracy=0.75,
        classifier_coverage=8 / 11,
        policy_switches=2,
        first_classification_hand=4,
        first_correct_classification_hand=5,
        final_predicted_type="calling",
    )

    df = pd.read_csv(
        output_path
    )

    assert len(df) == 1

    row = df.iloc[0]

    assert row["experiment_name"] == (
        "adaptive_mc_vs_calling"
    )
    assert row["agent_name"] == "adaptive_mc"
    assert row["opponent_name"] == "calling"
    assert row["final_stack"] == 0
    assert row["initial_stack"] == 200
    assert row["profit"] == -200
    assert row["profit_bb"] == -20
    assert row["won_game"] == 0
    assert row["busted"] == 1
    assert row["ended_by_bust"] == 1
    assert row["ended_by_round_limit"] == 0
    assert row["classified_decisions"] == 8
    assert row["correct_classifications"] == 6
    assert row["incorrect_classifications"] == 2
    assert row["unknown_classifications"] == 3
    assert row["classifier_accuracy"] == 0.75
    assert row["policy_switches"] == 2
    assert row["first_classification_hand"] == 4
    assert row["first_correct_classification_hand"] == 5
    assert row["final_predicted_type"] == "calling"


def test_result_logger_detects_winning_game(
    tmp_path,
):
    output_path = tmp_path / "results.csv"

    logger = ResultLogger(
        str(output_path)
    )

    logger.log_game(
        experiment_name="adaptive_mc_vs_fish",
        game_id=1,
        agent_name="adaptive_mc",
        opponent_name="fish",
        final_stack=400,
        initial_stack=200,
        hands_played=20,
        big_blind=10,
        ended_by_bust=True,
        ended_by_round_limit=False,
    )

    row = pd.read_csv(
        output_path
    ).iloc[0]

    assert row["profit"] == 200
    assert row["profit_bb"] == 20
    assert row["won_game"] == 1
    assert row["busted"] == 0
    assert row["ended_by_bust"] == 1


def test_result_logger_records_round_limit(
    tmp_path,
):
    output_path = tmp_path / "results.csv"

    logger = ResultLogger(
        str(output_path)
    )

    logger.log_game(
        experiment_name="rule_based_vs_calling",
        game_id=1,
        agent_name="rule_based",
        opponent_name="calling",
        final_stack=210,
        initial_stack=200,
        hands_played=100,
        big_blind=10,
        ended_by_bust=False,
        ended_by_round_limit=True,
    )

    row = pd.read_csv(
        output_path
    ).iloc[0]

    assert row["ended_by_bust"] == 0
    assert row["ended_by_round_limit"] == 1
    assert row["busted"] == 0


def test_result_logger_rejects_non_positive_big_blind(
    tmp_path,
):
    logger = ResultLogger(
        str(tmp_path / "results.csv")
    )

    with pytest.raises(
        ValueError,
        match="big_blind must be greater than zero",
    ):
        logger.log_game(
            experiment_name="test",
            game_id=1,
            agent_name="adaptive_mc",
            opponent_name="fish",
            final_stack=200,
            initial_stack=200,
            hands_played=10,
            big_blind=0,
            ended_by_bust=False,
            ended_by_round_limit=False,
        )


def test_result_logger_appends_multiple_rows(
    tmp_path,
):
    output_path = tmp_path / "results.csv"

    logger = ResultLogger(
        str(output_path)
    )

    for game_id in range(3):
        logger.log_game(
            experiment_name="test",
            game_id=game_id,
            agent_name="adaptive_mc",
            opponent_name="fish",
            final_stack=200,
            initial_stack=200,
            hands_played=10,
            big_blind=10,
            ended_by_bust=False,
            ended_by_round_limit=False,
        )

    df = pd.read_csv(
        output_path
    )

    assert len(df) == 3