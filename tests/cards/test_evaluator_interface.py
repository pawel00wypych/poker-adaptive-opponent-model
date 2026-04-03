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
    assert result == expected_result


def test_eval_hand_weak_preflop():
    hole = ["C9", "DT"]
    community = []
    expected_result = {
        'hand': {'strength': 'HIGHCARD','high': 10,'low': 9},
        'hole': {'high': 10,'low': 9},
        'score': 43433}
    result = EvaluatorInterface.evaluate(hole, community)
    assert result == expected_result

def test_eval_hand_weak_flop():
    hole = ["C9", "DT"]
    community = ["D2", "CA", "SJ"]
    expected_result = {
        'hand': {'strength': 'HIGHCARD','high': 10,'low': 9},
        'hole': {'high': 10,'low': 9},
        'score': 43433}
    result = EvaluatorInterface.evaluate(hole, community)
    print(result)
    assert result == expected_result

def test_eval_hand_medium_turn():
    hole = ["C2", "D3"]
    community = ["H3", "S2", "SJ", "ST"]
    expected_result = {
        'hand': {'strength': 'TWOPAIR', 'high': 3, 'low': 2},
        'hole': {'high': 3, 'low': 2},
        'score': 143922
    }
    result = EvaluatorInterface.evaluate(hole, community)
    print(result)
    assert result == expected_result

def test_eval_hand_medium_river():
    hole = ["C2", "D8"]
    community = ["H2", "S8", "SJ", "ST", "H9"]
    expected_result = {'hand': {'strength': 'TWOPAIR', 'high': 8, 'low': 2},
                       'hole': {'high': 8, 'low': 2},
                       'score': 164482
                       }
    result = EvaluatorInterface.evaluate(hole, community)
    print(result)
    assert result == expected_result