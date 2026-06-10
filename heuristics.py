from game_state import ChessGameState
import bulletchess
import random


def heuristic_probabilistic_rollout(state: ChessGameState) -> float:
    """
    A soft-attention rollout policy.
    Instead of strictly prioritizing forcing moves, it samples them
    probabilistically based on salience weights:
    - Check:   10.0
    - Capture: 3.0
    - Quiet:   1.0
    """
    steps = 0
    max_steps = 50
    curr = state

    while not curr.is_terminal() and steps < max_steps:
        actions = curr.get_legal_actions()
        if not actions:
            break

        weights = []
        for action in actions:
            if curr.gives_check(action):
                weights.append(10.0)
            elif curr.is_capture(action):
                weights.append(3.0)
            else:
                weights.append(1.0)

        # Weighted sampling
        action = random.choices(actions, weights=weights, k=1)[0]

        curr = curr.apply_action(action)
        steps += 1

    reward = curr.get_reward()

    # Flip reward to perspective of the parent node
    if steps % 2 == 1:
        reward = -reward

    return reward

def heuristic_chess_rollout(state: ChessGameState, max_depth=40):
    """
    Performs a heuristic-based rollout (Mate > Check > Capture > Random).
    Returns the reward relative to the player who JUST MOVED to reach
    the state (i.e., the Parent of this node).

    This ensures that if White plays a winning move, the node stores +1.0.
    """
    # 1. Identify the player to move at the start (The 'Child' player)
    child_turn = state.board.turn

    curr = state
    steps = 0

    # --- Simulation Phase ---
    while True:
        if curr.board in bulletchess.CHECKMATE or \
                curr.board in bulletchess.DRAW or \
                steps >= max_depth:
            break

        actions = curr.get_legal_actions()
        if not actions:
            break

        selected = None

        # Priority 1: Mate
        for m in actions:
            if curr.gives_mate(m):
                selected = m;
                break

        # Priority 2: Check
        if not selected:
            checks = []
            for move in actions:
                next_state = curr.apply_action(move)
                if next_state.board in bulletchess.CHECK:
                    checks.append(move)
            if checks: selected = random.choice(checks)

        # Priority 3: Capture
        if not selected:
            captures = [m for m in actions if curr.is_capture(m)]
            if captures: selected = random.choice(captures)

        # Priority 4: Random
        if not selected:
            selected = random.choice(actions)

        curr = curr.apply_action(selected)
        steps += 1

    # --- Scoring Phase ---

    if curr.board in bulletchess.CHECKMATE:
        # The player to move at 'curr' has been checkmated and LOSES.
        loser = curr.board.turn

        # Logic Flip:
        # If the 'Child' player (Black) is the loser, then the 'Parent' (White) Won.
        # We want to return value for the Parent.

        if loser == child_turn:
            return 1.0  # Parent Won
        else:
            return -1.0  # Parent Lost

    elif curr.board in bulletchess.DRAW:
        return 0.0

    return 0.0
