# Bayesian Memory MCTS

This is a compact research prototype for memory-augmented Monte Carlo Tree
Search in chess. It combines Bayesian action-value estimates, value of
information search control, Thompson sampling, and episodic memory traces.

The code here is the cleaned canonical subset of a larger exploratory workspace.
It is intended to be readable and runnable, not a general-purpose chess engine or
installable library.

## Core Idea

Each child node stores a Dirichlet posterior over three outcomes: win, draw, and
loss. The search derives an action value from that posterior as:

```text
Q = P(win) - P(loss)
```

Search can select actions by either:

- sampling from the Dirichlet posterior with Thompson sampling, or
- scoring actions by posterior mean plus a value-of-perfect-information term.

After search, expanded tree values are consolidated with a minimax backup pass.
This keeps computation allocation separate from final move choice: VPI can spend
rollouts where information is valuable, while the selected move is based on the
game-theoretic value currently represented in the tree.

Episodic memory initializes child-node outcome counts when the search revisits a
known position. Memory traces can also encode criticality and recency.

## Files

- `mcts.py`: Bayesian MCTS loop, VPI calculation, Thompson sampling, and solver
  propagation.
- `node.py`: Dirichlet node state and posterior updates.
- `memory.py`: episodic memory store with criticality, decay, and forgetting.
- `game_state.py`: `bulletchess` wrapper for the search-state interface.
- `heuristics.py`: rollout policies for chess.
- `experiment_runner.py`: small batch experiment over mate puzzles.
- `demo.py`: fast demonstration on a mate-in-3 position with injected memory.
- `tests/`: unit and integration tests.

## Setup

The local environment used during development is:

```bash
/opt/miniconda3/envs/chess13/bin/python
```

For a fresh environment:

```bash
python -m pip install -r requirements.txt
```

## Quick Run

```bash
python demo.py
```

Expected behavior: the model selects `h5g6` as the root move in the included
mate-in-3 position.

## Tests

From this directory:

```bash
python -m unittest discover -s tests
```

Current local baseline under `/opt/miniconda3/envs/chess13/bin/python`: all
tests pass.

## Experiment

```bash
python experiment_runner.py
```

The default settings are intentionally small so the script can run as a quick
smoke test. Increase `MEMORY_LEVELS`, `MAX_SIMS`, and `REPEATS` at the bottom of
`experiment_runner.py` for larger sweeps.

## Notes

This upload excludes older prototypes, debug scripts, generated visualizations,
notebooks, cached bytecode, and local IDE metadata.

## License

Licensed under GPL-3.0-or-later. See `LICENSE`.
