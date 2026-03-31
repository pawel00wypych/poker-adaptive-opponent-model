import pytest

@pytest.fixture
def valid_actions_conf():
    return [
              {'action': 'fold', 'amount': 0},
              {'action': 'call', 'amount': 0},
              {'action': 'raise', 'amount': {'max': 95, 'min': 20}}
            ]

