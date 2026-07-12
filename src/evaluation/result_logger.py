import csv
from pathlib import Path


class ResultLogger:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.fieldnames = [
            "experiment_name",
            "game_id",
            "agent_name",
            "final_stack",
            "initial_stack",
            "profit",
            "profit_bb",
            "hands_played",
            "won_game",
            "busted",
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
        final_stack: int,
        initial_stack: int,
        hands_played: int,
        big_blind: int,
    ) -> None:
        profit = final_stack - initial_stack
        profit_bb = profit / big_blind

        row = {
            "experiment_name": experiment_name,
            "game_id": game_id,
            "agent_name": agent_name,
            "final_stack": final_stack,
            "initial_stack": initial_stack,
            "profit": profit,
            "profit_bb": profit_bb,
            "hands_played": hands_played,
            "won_game": int(final_stack > initial_stack),
            "busted": int(final_stack == 0),
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