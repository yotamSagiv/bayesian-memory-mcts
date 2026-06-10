# node.py
import numpy as np
import math
from typing import Dict, Any, List, Optional


class DirichletNode:
    def __init__(self, parent, action_from_parent, prior_counts: List[float]):
        """
        Args:
            prior_counts: [Win, Draw, Loss] (e.g., [1.0, 0.1, 0.1] for optimism)
        """
        self.parent = parent
        self.action = action_from_parent
        self.children: Dict[Any, 'DirichletNode'] = {}
        self.is_expanded = False

        # Solver State
        # 1.0 (Win), -1.0 (Loss), 0.0 (Draw), or None (Unsolved)
        self.solved_outcome: Optional[float] = None
        self.total_legal_moves: int = 0

        # Store counts as float32 for efficiency
        self.counts = np.array(prior_counts, dtype=np.float32)
        self.consolidated_value: Optional[float] = None

    def force_certainty(self, outcome: float):
        """
        MCTS-Solver Logic:
        Collapses the distribution to a Dirac delta (Infinite confidence).
        """
        self.solved_outcome = outcome
        self.counts[:] = 0.0

        # Set a massive count to dominate any exploration term forever
        huge_count = 1000

        if outcome == 1.0:
            self.counts[0] = huge_count
        elif outcome == 0.0:
            self.counts[1] = huge_count
        elif outcome == -1.0:
            self.counts[2] = huge_count

    @property
    def total_strength(self) -> float:
        return np.sum(self.counts)

    @property
    def mu(self) -> float:
        """E[Q] = P(Win) - P(Loss)"""
        S = self.total_strength
        return (self.counts[0] - self.counts[2]) / S

    @property
    def sigma(self) -> float:
        """Dirichlet-Normal Approximation of Variance"""
        alpha_w, alpha_l = self.counts[0], self.counts[2]
        S = self.total_strength
        if S == 0: return 1.0

        var_w = (alpha_w * (S - alpha_w)) / (S ** 2 * (S + 1))
        var_l = (alpha_l * (S - alpha_l)) / (S ** 2 * (S + 1))
        cov_wl = (-alpha_w * alpha_l) / (S ** 2 * (S + 1))

        var_q = var_w + var_l - 2 * cov_wl
        return math.sqrt(max(0, var_q))

    def update(self, reward: float):
        """
        Updates counts based on discrete outcome.
        Strictly accepts 1.0 (Win), 0.0 (Draw), -1.0 (Loss).
        Raises ValueError otherwise.
        """
        if reward == 1.0:
            self.counts[0] += 1.0
        elif reward == 0.0:
            self.counts[1] += 1.0
        elif reward == -1.0:
            self.counts[2] += 1.0
        else:
            raise ValueError(f"Invalid reward: {reward}. Must be 1.0, 0.0, or -1.0")

    def integrate_memory_samples(self, samples: List[float]):
        """Ingests historical outcomes."""
        for outcome in samples:
            self.update(outcome)
