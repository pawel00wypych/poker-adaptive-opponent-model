from src.cards.evaluator_Interface import EvaluatorInterface


def test_eval_hand():
    hole = ["CA", "DK"]
    community = ["DA", "H6", "S6", "H5", "C4"]
    expected_result = {
        "hand": {"strength": "TWOPAIR", "high": 14, "low": 6},
        "hole": {"high": 14, "low": 13},
        "score": 190189,
    }
    result = EvaluatorInterface.evaluate(hole, community)
    assert result == expected_result


def test_eval_hand_community_empty():
    hole = ["CA", "DK"]
    community = []
    expected_result = {
        "hand": {"strength": "HIGHCARD", "high": 14, "low": 13},
        "hole": {"high": 14, "low": 13},
        "score": 60909,
    }
    result = EvaluatorInterface.evaluate(hole, community)
    result == expected_result