import sys

import pytest

from src.experiments.cli_utils import (
    parse_specialist_training_args,
    parse_training_args,
)


def test_training_args_have_expected_defaults(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["program"],
    )

    args = parse_training_args()

    assert args.progress is True
    assert args.player_verbose is False
    assert args.player_log_interval == 1
    assert args.engine_verbose is False
    assert args.log_interval == 100


def test_specialist_args_parse_opponent(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "program",
            "--opponent",
            "calling",
            "--log-interval",
            "500",
            "--player-log-interval",
            "10",
        ],
    )

    args = parse_specialist_training_args()

    assert args.opponent == "calling"
    assert args.log_interval == 500
    assert args.player_log_interval == 10


def test_specialist_args_reject_invalid_opponent(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "program",
            "--opponent",
            "balanced",
        ],
    )

    with pytest.raises(
        SystemExit,
    ):
        parse_specialist_training_args()


def test_training_args_reject_zero_log_interval(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "program",
            "--log-interval",
            "0",
        ],
    )

    with pytest.raises(
        SystemExit,
    ):
        parse_training_args()


def test_training_args_reject_zero_player_log_interval(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "program",
            "--player-log-interval",
            "0",
        ],
    )

    with pytest.raises(
        SystemExit,
    ):
        parse_training_args()

def test_training_args_parse_alpha_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "program",
            "--alpha-mode",
            "sqrt_visit",
        ],
    )

    args = parse_training_args()

    assert args.alpha_mode == "sqrt_visit"


def test_specialist_args_parse_alpha_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "program",
            "--opponent",
            "calling",
            "--alpha-mode",
            "visit_count",
        ],
    )

    args = parse_specialist_training_args()

    assert args.alpha_mode == "visit_count"
