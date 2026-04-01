
class EvaluatorInterface:
    def evaluate(self, hole, community) -> dict:
        """
        Returns at least:
        - 'score': integer, comparable for winner determination
        - 'hand': optional dict with 'strength', 'high', 'low', etc.
        """
        pass