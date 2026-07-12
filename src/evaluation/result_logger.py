import csv
from pathlib import Path


class ResultLogger:
    def __init__(
        self,
        output_path: str,
    ):
        self.output_path = Path(
            output_path
        )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.fieldnames = [
            "experiment_name",
            "game_id",
            "agent_name",
            "opponent_name",
            "final_stack",
            "initial_stack",
            "profit",
            "profit_bb",
            "hands_played",
            "won_game",
            "busted",
            "ended_by_bust",
            "ended_by_round_limit",
            "classified_decisions",
            "correct_classifications",
            "incorrect_classifications",
            "unknown_classifications",
            "classifier_accuracy",
            "classifier_coverage",
            "policy_switches",
            "first_classification_hand",
            "first_correct_classification_hand",
            "final_predicted_type",
        ]

        if not self.output_path.exists():
            with self.output_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=self.fieldnames,
                )
                writer.writeheader()

    def log_game(
        self,
        experiment_name: str,
        game_id: int,
        agent_name: str,
        opponent_name: str,
        final_stack: int,
        initial_stack: int,
        hands_played: int,
        big_blind: int,
        ended_by_bust: bool,
        ended_by_round_limit: bool,
        classified_decisions: int = 0,
        correct_classifications: int = 0,
        incorrect_classifications: int = 0,
        unknown_classifications: int = 0,
        classifier_accuracy: float = 0.0,
        classifier_coverage: float = 0.0,
        policy_switches: int = 0,
        first_classification_hand: int | None = None,
        first_correct_classification_hand: int | None = None,
        final_predicted_type: str = "",
    ) -> None:
        if big_blind <= 0:
            raise ValueError(
                "big_blind must be greater than zero"
            )

        profit = (
            final_stack - initial_stack
        )
        profit_bb = profit / big_blind

        row = {
            "experiment_name": experiment_name,
            "game_id": game_id,
            "agent_name": agent_name,
            "opponent_name": opponent_name,
            "final_stack": final_stack,
            "initial_stack": initial_stack,
            "profit": profit,
            "profit_bb": profit_bb,
            "hands_played": hands_played,
            "won_game": int(
                final_stack > initial_stack
            ),
            "busted": int(
                final_stack == 0
            ),
            "ended_by_bust": int(
                ended_by_bust
            ),
            "ended_by_round_limit": int(
                ended_by_round_limit
            ),
            "classified_decisions": (
                classified_decisions
            ),
            "correct_classifications": (
                correct_classifications
            ),
            "incorrect_classifications": (
                incorrect_classifications
            ),
            "unknown_classifications": (
                unknown_classifications
            ),
            "classifier_accuracy": (
                classifier_accuracy
            ),
            "classifier_coverage": (
                classifier_coverage
            ),
            "policy_switches": policy_switches,
            "first_classification_hand": (
                first_classification_hand
            ),
            "first_correct_classification_hand": (
                first_correct_classification_hand
            ),
            "final_predicted_type": (
                final_predicted_type
            ),
        }

        with self.output_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.fieldnames,
            )
            writer.writerow(row)