from pypokerengine.players import BasePokerPlayer
from src.cards.hand_estimator import HandEstimator
from src.cards.hand_estimator import HandStrength

class PlayerTemplate(BasePokerPlayer):
    """Poker player template

      Class contains basic statistics used for player evaluation.
      Other poker player classes inherit from this class.
      """

    def __init__(self,
                 player_name='',
                 roi=0,
                 hands_played=0,
                 win_rate=0,
                 total_profit=0,
                 vpip=0,
                 pfr=0,
                 vpip_pfr_gap=0,
                 three_bet=0,
                 fold_to_three_bet=0,
                 af=0,
                 afq=0,
                 cbet=0,
                 fold_to_cbet=0,
                 steal_attempt=0,
                 fold_to_steal=0,
                 win_rate_by_position=None

        ):
        """
        Volume & Results metrics:

        - hands_played
        total sample size

        - win_rate
        average profit in big blinds per 100 hands

        - total_profit
        net earnings over time

        - ROI (return on investment)
        ROI = profit / total buy-ins
        """
        super().__init__()
        self.player_name = str(self.uuid) if player_name == '' else player_name
        self.roi = roi
        self.hands_played = hands_played
        self.win_rate = win_rate
        self.total_profit = total_profit
        '''
        Preflop Statistics:
        
        - VPIP (Voluntarily Put Money in Pot)
        % of hands where player invests preflop (excluding blinds)
        measures looseness
        
        - PFR (Preflop Raise)
        % of hands where player raises preflop
        measures aggression
        
        - VPIP - PFR Gap
        Indicates passivity (large gap = calling a lot)
        
        - 3-Bet %
        % of times player re-raises preflop
        
        - Fold to 3-Bet %
        How often they fold after being re-raised
        '''
        self.vpip = vpip
        self.pfr = pfr
        self.vpip_pfr_gap = vpip_pfr_gap
        self.three_bet = three_bet
        self.fold_to_three_bet = fold_to_three_bet

        '''
        Postflop Aggression Metrics:
        
        - Aggression Factor (AF)
        AF = (Bets + Raises) / Calls
        
        - Aggression Frequency (AFq)
        % of aggressive actions out of all actions
        
        - Continuation Bet (C-Bet %)
        Flop C-Bet, Turn C-Bet
        
        - Fold to C-Bet %
        measures how easily player gives up
        '''
        self.AF = af
        self.AFq = afq
        self.CBet = cbet
        self.fold_to_cbet = fold_to_cbet

        '''
        Positional Statistics:
        
        - Steal Attempt %
        Raising from late position (CO, BTN, SB)
        
        - Fold to Steal %
        How often blinds are surrendered
        
        - Win Rate by Position
        (BTN usually highest, blinds lowest)
        '''

        self.steal_attempt = steal_attempt
        self.fold_to_steal = fold_to_steal
        if not win_rate_by_position:
            self.win_rate_by_position={
                         'SB': 0,
                         'BB': 0,
                         'MP': 0,
                         'BTN': 0
                     }
        else:
            self.win_rate_by_position = win_rate_by_position

        """
        Round related parameters
        """
        self.hole_card = []
        self.uuid_to_index = None
        self.my_index = None
        self.available_positions = {
                    "dealer_btn": None,
                    "small_blind_pos": None,
                    "big_blind_pos": None
        }
        self.my_position = self.available_positions.items()

        """
        Hand estimator
        """
        self.hand_estimator = HandEstimator()

    def set_my_position(self):
        position = [v for k,v in self.available_positions.items() if v ==
                    self.my_index]
        if len(position) > 1:
            self.my_position = "big_blind_pos" if "big_blind_pos" in position \
                else "small_blind_pos"

    def set_available_positions(self, round_state):
        self.available_positions["dealer_btn"] = round_state["dealer_btn"]
        self.available_positions["small_blind_pos"] = round_state["small_blind_pos"]
        self.available_positions["big_blind_pos"] = round_state["big_blind_pos"]



    """
    Derived methods without declare_action()
    """
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hole_card = hole_card
        self.uuid_to_index = {
            seat['uuid']: i for i, seat in enumerate(seats)
        }
        self.my_index = self.uuid_to_index[self.uuid]

    def receive_street_start_message(self, street, round_state):
        self.set_available_positions(round_state)
        self.set_my_position()

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        print(f"\n hand_info = {hand_info}")
        print(f"\n winners = {winners}")