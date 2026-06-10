import bulletchess

from game_state import ChessGameState
from heuristics import heuristic_chess_rollout
from mcts import BayesianMCTS
from memory import EpisodicMemoryStore


PUZZLE_FEN = "r4r2/1p5k/1p1ppqp1/2p1n2P/7Q/7R/PPP4P/6K1 w - - 0 28"
SOLUTION_MOVE = "h5g6"


def main():
    root_state = ChessGameState(fen=PUZZLE_FEN)
    solution_move = bulletchess.Move.from_uci(SOLUTION_MOVE)

    memory = EpisodicMemoryStore()
    memory.add(root_state.apply_action(solution_move), [1.0] * 5)

    mcts = BayesianMCTS(
        optimistic_prior=[1.0, 0.1, 0.1],
        memory=memory,
        rollout_policy=heuristic_chess_rollout,
        enable_solver=True,
        use_thompson_sampling=True,
    )

    best_move, simulations = mcts.search(
        root_state,
        max_simulations=100,
        vpi_threshold=0.01,
    )

    print(f"Puzzle: mate in 3 from {PUZZLE_FEN}")
    print(f"Expected root move: {SOLUTION_MOVE}")
    print(f"Selected root move: {best_move.uci() if best_move else 'None'}")
    print(f"Search simulations: {simulations}")


if __name__ == "__main__":
    main()
