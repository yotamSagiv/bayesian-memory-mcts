# Bayesian Memory MCTS for Chess

This repo contains code for a project that integrates (episodic) memory into planning.
The goals of this model are:

1) Improve search efficiency by using memory to bias search to historically-promising areas
of the solution landscape.
2) In accordance with empirical data on human chess play, reduce move times in positions
with high previous experience.

Broadly speaking, the way this occurs by initializing the value of a node, once it is expanded
based on historical value outcomes in similar positions. Unlike traditional MCTS, here
search is terminated on the basis of expected value of computation; when the next rollout
is not expected to be very valuable, search ends. 

In particular, the current implementation combines:

- Bayesian action values represented as Dirichlet counts over win/draw/loss.
- Value-of-information search control using a truncated-normal approximation.
- Optional Thompson sampling over the Dirichlet posterior.
- Episodic memory traces that can encode outcome, recency, and criticality.
- Lightweight MCTS-solver propagation for proven terminal outcomes.

## Main Files

- `mcts.py`: Bayesian MCTS search loop, VPI calculation, Thompson sampling, and
  solver propagation.
- `node.py`: Dirichlet node state, posterior mean/variance, count updates, and
  certainty forcing.
- `memory.py`: episodic memory store with criticality, decay, and forgetting.
- `game_state.py`: `bulletchess` wrapper implementing the search-state API.
- `heuristics.py`: chess rollout policies used by experiments.
- `experiment_runner.py`: batch experiments over mate puzzles and memory levels.

## Environment

Install dependencies in a fresh environment with:

```bash
python -m pip install -r requirements.txt
```

`bulletchess` is required for the chess wrapper and experiments.

## Main Experiment

Run the batch mate-puzzle experiment with experiment_runner.py.

The default configuration runs many repeats and may take a while. For a quick
demo, reduce `REPEATS`, `MAX_SIMS`, or `MEMORY_LEVELS` at the bottom of
`experiment_runner.py`.