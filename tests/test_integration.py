import unittest

import bulletchess

from mcts import BayesianMCTS
from memory import EpisodicMemoryStore
from game_state import ChessGameState


class TwoMoveChessState(ChessGameState):
    def __init__(self, board=None, fen=None, allowed_moves=None):
        super().__init__(board, fen)
        self.allowed_moves = allowed_moves if allowed_moves else []

    def get_legal_actions(self):
        all_moves = super().get_legal_actions()
        allowed_uci = {m.uci() for m in self.allowed_moves}
        return [m for m in all_moves if m.uci() in allowed_uci]


class TestMemoryBias(unittest.TestCase):
    def setUp(self):
        self.start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.good_move = bulletchess.Move.from_uci("e2e4")
        self.bad_move = bulletchess.Move.from_uci("h2h3")

    def rigged_rollout(self, state):
        history = state.board.history
        if not history:
            return 0.0

        root_move = history[0]
        if root_move.uci() == "e2e4":
            global_score = 1.0
        elif root_move.uci() == "h2h3":
            global_score = -1.0
        else:
            global_score = 0.0

        if state.board.turn == bulletchess.WHITE:
            return -global_score
        return global_score

    def state_after(self, move):
        board = bulletchess.Board.from_fen(self.start_fen)
        board.apply(move)
        return ChessGameState(fen=board.fen())

    def test_memory_bias_selects_known_good_move(self):
        root_state = TwoMoveChessState(
            fen=self.start_fen,
            allowed_moves=[self.good_move, self.bad_move],
        )

        memory = EpisodicMemoryStore()
        memory.add(self.state_after(self.good_move), [1.0] * 5)
        memory.add(self.state_after(self.bad_move), [-1.0] * 5)

        mcts = BayesianMCTS(
            optimistic_prior=[1.0, 0.1, 0.1],
            memory=memory,
            rollout_policy=self.rigged_rollout,
        )

        best_move, simulations = mcts.search(root_state, max_simulations=1000, vpi_threshold=0.01)

        self.assertEqual(best_move.uci(), "e2e4")
        self.assertLessEqual(simulations, 2)


if __name__ == "__main__":
    unittest.main()
