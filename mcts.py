import numpy as np
import math
from typing import Callable, Optional, Any
from node import DirichletNode
from game_state import AbstractGameState
from memory import EpisodicMemoryStore
import random

# --- Optimization Helpers (Replaces scipy.stats for speed) ---
SQRT_2 = math.sqrt(2)
SQRT_2PI = math.sqrt(2 * math.pi)

def _fast_pdf(x):
    """Standard Normal PDF."""
    return math.exp(-0.5 * x**2) / SQRT_2PI

def _fast_cdf(x):
    """Standard Normal CDF using Error Function."""
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


def default_random_rollout(state: AbstractGameState) -> float:
    """
    A generic random walk policy.
    Rolls out up to 50 steps or until terminal.
    """
    steps = 0
    max_steps = 50
    curr = state

    while not curr.is_terminal() and steps < max_steps:
        actions = curr.get_legal_actions()
        if not actions:
            break
        action = random.choice(actions)
        curr = curr.apply_action(action)
        steps += 1

    reward = curr.get_reward()

    # Convert the terminal reward to the Parent's perspective.
    if steps % 2 == 1:
        reward = -reward

    return reward


class BayesianMCTS:
    def __init__(self,
                 optimistic_prior=(1.0, 0.1, 0.1),
                 memory: Optional[EpisodicMemoryStore] = None,
                 rollout_policy: Optional[Callable[[AbstractGameState], float]] = None,
                 enable_solver: bool = False,
                 use_thompson_sampling: bool = False):
        """
        Args:
            optimistic_prior: [Win, Draw, Loss] counts for initialization.
            memory: Optional EpisodicMemoryStore for initializing priors.
            rollout_policy: Function taking a state and returning a reward float.
        """
        if any(p <= 0 for p in optimistic_prior):
            raise ValueError("Priors must be strictly positive.")

        self.prior = list(optimistic_prior)
        self.memory = memory
        self.rollout_policy = rollout_policy if rollout_policy else default_random_rollout
        self.enable_solver = enable_solver
        self.root: Optional[DirichletNode] = None
        self.use_thompson_sampling = use_thompson_sampling

    def reset(self):
        self.root = None

    def update_root(self, action: Any, state: AbstractGameState):
        if self.root is not None and self.root.is_expanded and action in self.root.children:
            new_root = self.root.children[action]
            new_root.parent = None
            self.root = new_root
        else:
            self.reset()

    def calculate_vpi(self, node: DirichletNode, best_alt_mu: float) -> float:
        """
        Calculates VPI using a Truncated Normal distribution [-1, 1].
        Handles Challenger (Right Tail) and Incumbent (Left Tail) logic.
        """
        sigma = node.sigma
        mu = node.mu

        if sigma < 1e-9:
            return 0.0

        # 1. Define Truncation Bounds
        a, b = -1.0, 1.0

        # 2. Calculate Normalization Constant (Z)
        # This scales the PDF so the area within [-1, 1] equals 1.0
        z_a = (a - mu) / sigma
        z_b = (b - mu) / sigma

        # Use fast math helpers
        Z = _fast_cdf(z_b) - _fast_cdf(z_a)

        if Z < 1e-12:
            return 0.0

        # --- CASE 1: The Challenger (Standard VPI) ---
        # We hope this node is BETTER than the alternative (Right Tail).
        if mu <= best_alt_mu:
            lower = max(best_alt_mu, a)
            upper = b

            if lower >= upper:
                return 0.0

            z_lower = (lower - mu) / sigma
            z_upper = (upper - mu) / sigma

            pdf_diff = _fast_pdf(z_lower) - _fast_pdf(z_upper)
            cdf_diff = _fast_cdf(z_upper) - _fast_cdf(z_lower)

            term1 = sigma * pdf_diff / Z
            term2 = (mu - best_alt_mu) * cdf_diff / Z
            return max(0.0, term1 + term2)

        # --- CASE 2: The Incumbent (Risk Analysis) ---
        # We worry this node is WORSE than the alternative (Left Tail).
        else:
            lower = a
            upper = min(best_alt_mu, b)

            if lower >= upper:
                return 0.0

            z_lower = (lower - mu) / sigma
            z_upper = (upper - mu) / sigma

            # Note: PDF diff direction flips here compared to Case 1 due to integration by parts on (-x).
            pdf_diff = _fast_pdf(z_upper) - _fast_pdf(z_lower)
            cdf_diff = _fast_cdf(z_upper) - _fast_cdf(z_lower)

            term1 = (best_alt_mu - mu) * cdf_diff / Z
            term2 = sigma * pdf_diff / Z
            return max(0.0, term1 + term2)

    def _select(self, node: DirichletNode) -> DirichletNode:
        """
        Selects the next child to explore.
        Modes:
          1. Thompson Sampling: Samples from posterior Dirichlet.
          2. Q + VPI: Uses Mean value + Value of Perfect Information.
        """
        # --- MODE 1: THOMPSON SAMPLING ---
        if self.use_thompson_sampling:
            best_child = None
            best_sample_val = -float('inf')

            for child in node.children.values():
                # Safety: Solver logic might set counts to 0.0.
                # Dirichlet requires strictly positive alphas.
                # We take max(counts, epsilon) to ensure stability.
                safe_counts = np.maximum(child.counts, 1e-6)

                # 1. Sample vector [p_win, p_draw, p_loss]
                sample = np.random.dirichlet(safe_counts)

                # 2. Calculate Q-Value of sample (Action Value logic)
                # Index 0 = Win for Parent, Index 2 = Loss for Parent
                val = sample[0] - sample[2]

                if val > best_sample_val:
                    best_sample_val = val
                    best_child = child

            return best_child

        # --- MODE 2: Q + VPI (Dearden) ---
        else:
            children = list(node.children.values())

            # Optimization: Only calculate VPI for top candidates
            sorted_by_mu = sorted(children, key=lambda c: c.mu, reverse=True)
            best_mu = sorted_by_mu[0].mu
            second_best_mu = sorted_by_mu[1].mu if len(children) > 1 else -1.0

            best_child = None
            best_score = -float('inf')

            for child in children:
                alt_mu = second_best_mu if child.mu == best_mu else best_mu
                vpi = self.calculate_vpi(child, alt_mu)

                score = child.mu + vpi

                if score > best_score:
                    best_score = score
                    best_child = child

            return best_child

    def _expand(self, node: DirichletNode, state: AbstractGameState):
        if node.is_expanded: return

        certainty_boost = 1_000.0
        actions = state.get_legal_actions()
        node.total_legal_moves = len(actions)

        for action in actions:
            child = DirichletNode(node, action, self.prior)
            next_state = state.apply_action(action)

            if self.memory:
                history = self.memory.retrieve(next_state)
                if history:
                    child.integrate_memory_samples(history)

            if next_state.is_terminal():
                reward = next_state.get_reward()
                child.counts[:] = 0.0

                # We store the Action Value (Q) for the Parent.
                # If Parent Won (+1), Child Node stores Win (+1).
                if reward > 0:
                    child.counts[0] = certainty_boost
                elif reward < 0:
                    child.counts[2] = certainty_boost
                else:
                    child.counts[1] = certainty_boost

                # Trigger Solver if enabled
                if self.enable_solver:
                    # Pass the Q-value directly
                    self._propagate_solve_status(child, reward)

            node.children[action] = child
        node.is_expanded = True

    def _propagate_solve_status(self, node: DirichletNode, outcome: float):
        """
        outcome: The Q-value of 'node' (Value for the Parent who chose it).
        """
        node.force_certainty(outcome)

        if node.parent is None:
            return

        # --- RECURSION LOGIC ---
        # If 'node' is a WIN (+1.0) for Parent, then Parent has found a winning move.
        # This means the Parent Node itself (which represents the PREVIOUS action)
        # is a LOSS (-1.0) for the Grandparent.

        val_for_grandparent = -outcome

        # CASE A: The "OR" Condition (Winning Move Found)
        # If I found a move that is a Win (+1.0) for me...
        if outcome == 1.0:
            # ...then I am Solved. My value to my parent (Grandparent) is -1.0.
            if node.parent.solved_outcome != val_for_grandparent:
                self._propagate_solve_status(node.parent, val_for_grandparent)
            return

        # CASE B: The "AND" Condition (All Moves Bad)
        # If 'node' is a Loss (-1.0) for Parent, Parent isn't screwed yet.
        # Parent waits until ALL children are proven Losses.
        if len(node.parent.children) == node.parent.total_legal_moves:
            all_solved = True
            best_outcome_for_parent = -1.0  # Assume worst (Loss)

            for sibling in node.parent.children.values():
                if sibling.solved_outcome is None:
                    all_solved = False
                    break

                # If any sibling is a Draw (0.0), Parent can force a Draw.
                if sibling.solved_outcome == 0.0:
                    best_outcome_for_parent = 0.0

                # Note: If a sibling was 1.0, CASE A would have run already.

            if all_solved:
                # Parent is solved. Propagate value to Grandparent.
                # If Parent outcome is -1.0 (Loss), Grandparent sees +1.0 (Win).
                self._propagate_solve_status(node.parent, -best_outcome_for_parent)

    def _simulate(self, state: AbstractGameState) -> float:
        return self.rollout_policy(state)

    def _backpropagate(self, node: DirichletNode, reward: float):
        curr = node
        val = reward
        while curr:
            # Block updates to solved nodes
            if self.enable_solver and curr.solved_outcome is not None:
                pass
            else:
                curr.update(val)
            curr = curr.parent
            val = -val

    def _consolidate_minimax(self, node: DirichletNode) -> float:
        if node.solved_outcome is not None:
            node.consolidated_value = node.solved_outcome
            return node.consolidated_value

        if not node.is_expanded or not node.children:
            node.consolidated_value = node.mu
            return node.consolidated_value

        best_child_value = max(self._consolidate_minimax(child) for child in node.children.values())
        node.consolidated_value = -best_child_value
        return node.consolidated_value

    def _best_root_child(self) -> DirichletNode:
        for child in self.root.children.values():
            self._consolidate_minimax(child)

        return max(
            self.root.children.values(),
            key=lambda child: (child.consolidated_value, child.total_strength)
        )

    def _print_debug_stats(self, step: int, root: DirichletNode, solution_uci: str):
        # 1. Find the Solution Node
        sol_node = None
        for action, child in root.children.items():
            if action.uci() == solution_uci:
                sol_node = child
                break

        # 2. Find the Current "Best" Node (The one MCTS would pick right now)
        # Sort by Mean (Mu) because that's what we select on
        sorted_children = sorted(root.children.values(), key=lambda c: c.mu, reverse=True)
        best_node = sorted_children[0] if sorted_children else None

        # 3. Extract Stats
        # Solution Stats
        if sol_node:
            s_mu = f"{sol_node.mu:.3f}"
            s_n = f"{sol_node.total_strength:.1f}"
            s_uc = f"{sol_node.counts}"
        else:
            s_mu, s_n, s_uc = "N/A", "0", "[]"

        # Best Node Stats
        if best_node:
            b_uci = best_node.action.uci()
            b_mu = f"{best_node.mu:.3f}"
            b_n = f"{best_node.total_strength:.1f}"
        else:
            b_uci, b_mu, b_n = "None", "0.0", "0"

        # 4. Principal Variation (PV) - "What is the agent thinking?"
        # Traverse down the best path for depth 3
        pv_line = []
        curr = best_node
        for _ in range(3):
            if curr and curr.children:
                # Greedy pick next best
                curr = max(curr.children.values(), key=lambda c: c.mu)
                pv_line.append(curr.action.uci())
            else:
                break
        pv_str = " ".join(pv_line)

        print(
            f"{step:<5} | SOL: {s_mu:>5} (N={s_n:>5}) {s_uc} | BEST: {b_uci:<5} {b_mu:>5} (N={b_n:>5}) | PV: {b_uci} {pv_str}")

    def search(self, root_state: AbstractGameState, max_simulations: int, vpi_threshold: float,
               debug_interval: int = 0, solution_uci: str = None):
        if root_state.is_terminal():
            return None, 0

        if self.root is None:
            self.root = DirichletNode(None, None, self.prior)

        self._expand(self.root, root_state)

        if debug_interval > 0:
            print(f"\n{'Step':<5} | {'SOLUTION STATS':<25} | {'CURRENT BEST STATS':<25} | {'PRINCIPAL VARIATION'}")
            print("-" * 90)

        if not self.root.children:
            return None, 0

        sims = 0
        termination_reason = "MAX_SIMULATIONS"

        for _ in range(max_simulations):
            if debug_interval > 0 and sims % debug_interval == 0:
                self._print_debug_stats(sims, self.root, solution_uci)

            # --- 1. SOLVER CHECK ---
            if self.enable_solver and self.root.solved_outcome is not None:
                termination_reason = f"SOLVER_LOGIC (Outcome={self.root.solved_outcome})"
                break

            # --- 2. VPI TERMINATION CHECK (Checking Priors) ---
            # Removed 'sims > 0' check. Now we check immediately.
            if self.root.is_expanded and len(self.root.children) > 1:
                children = list(self.root.children.values())
                sorted_c = sorted(children, key=lambda c: c.mu, reverse=True)

                max_vpi = 0.0
                for c in children:
                    alt = sorted_c[1].mu if c == sorted_c[0] else sorted_c[0].mu
                    vpi = self.calculate_vpi(c, alt)
                    if vpi > max_vpi: max_vpi = vpi

                if sims == 0 and debug_interval > 0:
                    print(f"   [PRIOR VPI] {max_vpi:.6f} (Threshold: {vpi_threshold})")

                if max_vpi < vpi_threshold:
                    termination_reason = f"VPI_CONVERGENCE (VPI={max_vpi:.6f})"
                    break

            # ... Standard MCTS Loop ...
            node = self.root
            state = root_state

            while node.is_expanded and node.children:
                node = self._select(node)
                state = state.apply_action(node.action)

            if not state.is_terminal():
                self._expand(node, state)

            r = self._simulate(state)
            self._backpropagate(node, r)
            sims += 1

        if debug_interval > 0:
            self._print_debug_stats(sims, self.root, solution_uci)

        best_child = self._best_root_child()

        if debug_interval > 0:
            print(f"   [STOP REASON] {termination_reason}")

        return best_child.action, sims
