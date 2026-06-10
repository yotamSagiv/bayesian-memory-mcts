# experiment_runner.py
import numpy as np
import time
import bulletchess
from dataclasses import dataclass
from typing import List, Literal, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

from mcts import BayesianMCTS
from memory import EpisodicMemoryStore
from game_state import ChessGameState
from heuristics import heuristic_chess_rollout


# --- DATA STRUCTURES ---

@dataclass
class Puzzle:
    name: str
    fen: str
    solution_line: List[str]

    @property
    def root_move(self) -> str:
        return self.solution_line[0]


@dataclass
class ExperimentConfig:
    name: str
    enable_solver: bool
    use_thompson: bool
    vpi_threshold: float
    color: str = "blue"
    marker: str = "o"


# --- MEMORY INJECTION LOGIC ---

def inject_memory_trace(memory: EpisodicMemoryStore,
                        root_state: ChessGameState,
                        puzzle: Puzzle,
                        n_count: int,
                        mode: Literal['root', 'path']):
    curr_state = root_state
    moves_to_inject = puzzle.solution_line if mode == 'path' else [puzzle.root_move]

    for move_uci in moves_to_inject:
        move = bulletchess.Move.from_uci(move_uci)
        curr_state = curr_state.apply_action(move)
        # Inject [1.0] (Win/Good Move) for n times
        memory.add(curr_state, [1.0] * n_count)


# --- EXPERIMENT ENGINE ---

def run_single_trial(puzzle: Puzzle,
                     config: ExperimentConfig,
                     n_memories: int,
                     max_sims: int,
                     memory_mode: str) -> Dict:
    """Runs one game/search and returns stats."""

    root_state = ChessGameState(fen=puzzle.fen)

    # 1. Setup Memory
    memory = EpisodicMemoryStore()
    if n_memories > 0:
        inject_memory_trace(memory, root_state, puzzle, n_memories, mode=memory_mode)

    # 2. Setup MCTS
    mcts = BayesianMCTS(
        optimistic_prior=[1.0, 0.1, 0.1],  # [Win, Draw, Loss]
        memory=memory,
        rollout_policy=heuristic_chess_rollout,
        enable_solver=config.enable_solver,
        use_thompson_sampling=config.use_thompson
    )

    # 3. Run Search
    start_time = time.time()
    best_move, sims = mcts.search(root_state, max_sims, config.vpi_threshold, debug_interval=0)
    elapsed = time.time() - start_time

    # 4. Evaluate
    move_uci = best_move.uci() if best_move else "None"
    success = (move_uci == puzzle.root_move)

    is_proven = False
    if mcts.root and mcts.root.solved_outcome is not None:
        is_proven = True

    return {
        "success": success,
        "sims": sims,
        "time": elapsed,
        "move": move_uci,
        "proven": is_proven
    }


def run_experiment_suite(puzzles: List[Puzzle],
                         configs: List[ExperimentConfig],
                         memory_levels: List[int],
                         max_simulations: int,
                         repeats: int = 5,
                         memory_mode: Literal['root', 'path'] = 'root') -> List[Dict]:
    results = []

    print(f"\n==================================================================================")
    print(f"   EXPERIMENT SUITE | Mode: {memory_mode.upper()} Injection | Repeats: {repeats}")
    print(f"==================================================================================")

    for puzzle in puzzles:
        print(f"\n>> PUZZLE: {puzzle.name}")
        print(f"   FEN: {puzzle.fen}")
        print(f"   PV : {puzzle.solution_line}")

        for config in configs:
            print(f"\n   [Config: {config.name}] (Thresh={config.vpi_threshold})")
            print(f"   {'Mem_N':<6} | {'Succ%':<6} | {'AvgSims':<9} | {'StdSims':<9} | {'AvgTime':<8} | {'Proven%'}")
            print(f"   {'-' * 65}")

            for n in memory_levels:
                trial_sims = []
                trial_times = []
                success_count = 0
                proven_count = 0

                for _ in range(repeats):
                    res = run_single_trial(puzzle, config, n, max_simulations, memory_mode)
                    trial_sims.append(res['sims'])
                    trial_times.append(res['time'])
                    if res['success']: success_count += 1
                    if res['proven']: proven_count += 1

                avg_sims = np.mean(trial_sims)
                std_sims = np.std(trial_sims)
                avg_time = np.mean(trial_times)
                succ_rate = (success_count / repeats) * 100
                prov_rate = (proven_count / repeats) * 100

                results.append({
                    "puzzle": puzzle.name,
                    "config": config.name,
                    "color": config.color,
                    "marker": config.marker,
                    "n": n,
                    "avg_sims": avg_sims,
                    "std_sims": std_sims,
                    "success_rate": succ_rate
                })

                print(
                    f"   {n:<6} | {succ_rate:5.1f}% | {avg_sims:9.1f} | {std_sims:9.1f} | {avg_time:8.4f} | {prov_rate:5.1f}%")

    return results


# --- PLOTTING FUNCTION ---

def plot_results(data: List[Dict], memory_mode: str):
    """
    Generates a scientific plot of the results with independent Y-axes.
    """
    try:
        sns.set_theme(style="whitegrid")
        sns.set_context("paper", font_scale=1.2)
    except:
        plt.style.use('ggplot')

    unique_puzzles = list(dict.fromkeys([d['puzzle'] for d in data]))
    unique_configs = list(dict.fromkeys([d['config'] for d in data]))

    n_cols = len(unique_puzzles)

    # CHANGE: sharey=False allows independent scaling for complex vs simple puzzles
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharey=False)

    if n_cols == 1: axes = [axes]

    for ax, puzzle_name in zip(axes, unique_puzzles):
        ax.set_title(puzzle_name, fontweight='bold')
        ax.set_xlabel("Memory Strength (N)")
        ax.set_ylabel("Simulations")  # Label on every plot now

        p_data = [d for d in data if d['puzzle'] == puzzle_name]

        for config_name in unique_configs:
            c_data = [d for d in p_data if d['config'] == config_name]
            if not c_data: continue

            c_data.sort(key=lambda x: x['n'])

            x = [d['n'] for d in c_data]
            y = [d['avg_sims'] for d in c_data]
            y_err = [d['std_sims'] for d in c_data]

            color = c_data[0]['color']
            marker = c_data[0]['marker']

            ax.plot(x, y, marker=marker, label=config_name, color=color, linewidth=2, markersize=6)

            lower = np.array(y) - np.array(y_err)
            upper = np.array(y) + np.array(y_err)
            lower = np.maximum(lower, 0)

            ax.fill_between(x, lower, upper, color=color, alpha=0.15)

    plt.suptitle(f"Transition from Search to Retrieval ({memory_mode.upper()} Injection)", y=1.05, fontsize=16)
    axes[0].legend()
    plt.tight_layout()
    plt.show()


# --- MAIN ---

if __name__ == "__main__":
    puzzles = [
        Puzzle("Mate in 1", "7r/ppp2Q1p/3kp3/3qp1n1/8/N1P5/PP3P1P/R4RK1 b - - 2 17", ["g5h3"]),
        Puzzle("Mate in 2", "8/5R1p/4p2k/p2p2p1/1p6/8/P1r2N1r/4RK2 w - - 0 37", ["e1e6", "h6h5", "f7h7"]),
        Puzzle("Mate in 3", "r4r2/1p5k/1p1ppqp1/2p1n2P/7Q/7R/PPP4P/6K1 w - - 0 28",
               ["h5g6", "h7g6", "h4h5", "g6g7", "h5h7"]),
        Puzzle("Mate in 4", "6bk/1Q6/P6p/2p3pP/2Pb4/3B2PK/5r2/8 b - - 1 42",
               ["g5g4", "h3h4", "d4f6", "h4g4", "g8e6", "d3f5", "e6f5"])
    ]

    configs = [
        # ExperimentConfig(
        #     name="Standard (VPI)",
        #     enable_solver=False,
        #     use_thompson=False,
        #     vpi_threshold=0.01,
        #     color="#E24A33",
        #     marker="s"
        # ),
        # ExperimentConfig(
        #     name="Thompson (Prob. Match)",
        #     enable_solver=False,
        #     use_thompson=True,
        #     vpi_threshold=0.001,
        #     color="#348ABD",
        #     marker="o"
        # ),
        ExperimentConfig(
            name="Solver + Thompson",
            enable_solver=True,
            use_thompson=True,
            vpi_threshold=0.001,
            color="#988ED5",
            marker="^"
        )
    ]

    # Small defaults keep the script suitable as a quick smoke run. Increase
    # these values for full experimental sweeps.
    MEMORY_LEVELS = [0, 5, 20]
    MAX_SIMS = 1000
    REPEATS = 3

    results = run_experiment_suite(
        puzzles=puzzles,
        configs=configs,
        memory_levels=MEMORY_LEVELS,
        max_simulations=MAX_SIMS,
        repeats=REPEATS,
        memory_mode='root'
    )

    plot_results(results, memory_mode='root')
