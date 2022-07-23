from rlgym.utils.gamestates import GameState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition


class KickoffTimeoutCondition(TimeoutCondition):
    def is_terminal(self, current_state: GameState):
        if current_state.ball.position[0] == 0 and current_state.ball.position[1] == 0:
            return False
        else:
            return super(KickoffTimeoutCondition, self).is_terminal(current_state)
