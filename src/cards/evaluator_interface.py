from pypokerengine.engine.card import Card
from pypokerengine.engine.hand_evaluator import HandEvaluator


class EvaluatorInterface:
    @classmethod
    def evaluate(cls, hole, community) -> dict:
        """
        Returns at least:
        - 'score': integer, comparable for winner determination
        - 'hand': optional dict with 'strength', 'high', 'low', etc.
        """
        hole = [cls.decode_str_to_card(c) for c in hole]
        hand = [cls.decode_str_to_card(c) for c in community]
        # HandEvaluator.eval_hand() returns integer in this
        # format:
        # [ hand strength bits ][ hand rank high ][ hand rank low ]
    # [ hole high ][ hole low ]
        score = HandEvaluator.eval_hand(hole, hand)
        hand_info = HandEvaluator.gen_hand_rank_info(hole, hand)
        hand_info['score'] = score
        return hand_info

    @classmethod
    def decode_str_to_card(cls, card: str) -> Card:
        return Card.from_str(card)

