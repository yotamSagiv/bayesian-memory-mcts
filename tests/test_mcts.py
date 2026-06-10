import unittest
import numpy as np
from typing import List, Any
from mcts import BayesianMCTS, DirichletNode
from game_state import AbstractGameState


# --- MOCK CLASSES FOR TESTING ---

class MockGameState(AbstractGameState):
    def __init__(self, name="root", legal_actions=None, is_terminal=False, reward=0.0):
        self.name = name
        self._legal_actions = legal_actions if legal_actions else []
        self._is_terminal = is_terminal
        self._reward = reward

    def get_legal_actions(self) -> List[Any]:
        return self._legal_actions

    def apply_action(self, action: Any) -> 'MockGameState':
        return action

    def is_terminal(self) -> bool:
        return self._is_terminal

    def get_reward(self) -> float:
        return self._reward

    def is_capture(self, a): return False

    def gives_check(self, a): return False

    def gives_mate(self, a): return False

    def __repr__(self): return f"State({self.name})"


class TestDirichletNode(unittest.TestCase):
    def setUp(self):
        self.prior = [1.0, 0.1, 0.1]
        self.node = DirichletNode(None, "root_action", self.prior)

    def test_initialization(self):
        """Test that priors are set correctly."""
        np.testing.assert_array_almost_equal(self.node.counts, np.array([1.0, 0.1, 0.1], dtype=np.float32))
        self.assertAlmostEqual(self.node.total_strength, 1.2, places=5)

        # Mu = (Win - Loss) / Total
        expected_mu = (1.0 - 0.1) / 1.2
        self.assertAlmostEqual(self.node.mu, expected_mu, places=5)

    def test_update_logic(self):
        self.node.update(1.0)
        self.assertEqual(self.node.counts[0], 2.0)
        self.node.update(-1.0)
        self.assertEqual(self.node.counts[2], 1.1)
        self.node.update(0.0)
        self.assertEqual(self.node.counts[1], 1.1)
        with self.assertRaises(ValueError):
            self.node.update(0.5)

    def test_memory_integration(self):
        samples = [1.0, 1.0, -1.0]
        self.node.integrate_memory_samples(samples)
        self.assertEqual(self.node.counts[0], 3.0)
        self.assertEqual(self.node.counts[2], 1.1)

    def test_force_certainty(self):
        """Test deterministic solver outcomes."""
        self.node.force_certainty(-1.0)

        self.assertEqual(self.node.counts[2], 1_000.0)

        self.assertEqual(self.node.counts[0], 0.0)
        self.assertEqual(self.node.solved_outcome, -1.0)


class TestBayesianMCTS(unittest.TestCase):
    def setUp(self):
        self.mcts = BayesianMCTS(enable_solver=True)

    def test_expand_action_value_check(self):
        """
        Scenario: Root -> Action 'WinMove' -> Terminal State (Reward=1.0).
        ACTION VALUE LOGIC:
            - Reward = 1.0 (Parent Won).
            - Child Node stores "Value for Parent".
            - Child Node should record a WIN (Index 0).
        """
        terminal_win = MockGameState(name="term", is_terminal=True, reward=1.0)
        root_state = MockGameState(name="root", legal_actions=[terminal_win])

        root_node = DirichletNode(None, None, self.mcts.prior)
        self.mcts._expand(root_node, root_state)

        child = root_node.children[terminal_win]

        # Expect High Wins (Index 0) because it's a good move for the parent
        self.assertGreater(child.counts[0], 900.0, "Child should record a Win (Index 0)")
        self.assertEqual(child.solved_outcome, 1.0)

    def test_expand_draw_perspective(self):
        terminal_draw = MockGameState(name="draw", is_terminal=True, reward=0.0)
        root_state = MockGameState(name="root", legal_actions=[terminal_draw])

        root_node = DirichletNode(None, None, self.mcts.prior)
        self.mcts._expand(root_node, root_state)

        child = root_node.children[terminal_draw]
        self.assertGreater(child.counts[1], 900.0, "Child should record Draw (Index 1)")
        self.assertEqual(child.solved_outcome, 0.0)

    def test_backpropagate_negamax(self):
        root = DirichletNode(None, None, self.mcts.prior)
        child = DirichletNode(root, "act1", self.mcts.prior)
        grandchild = DirichletNode(child, "act2", self.mcts.prior)

        # Grandchild action returns +1.0
        self.mcts._backpropagate(grandchild, 1.0)

        # Grandchild node stores +1.0
        self.assertEqual(grandchild.counts[0], 2.0)
        # Child node stores -1.0 (Loss for child player)
        self.assertAlmostEqual(child.counts[2], 1.1, places=5)
        # Root node stores +1.0 (Win for root player)
        self.assertEqual(root.counts[0], 2.0)

    def test_vpi_calculation(self):
        node_certain = DirichletNode(None, None, [100, 0, 0])
        vpi = self.mcts.calculate_vpi(node_certain, -1.0)
        self.assertAlmostEqual(vpi, 0.0, places=5)

        node_uncertain = DirichletNode(None, None, [1, 1, 1])
        vpi_high = self.mcts.calculate_vpi(node_uncertain, 0.1)
        self.assertGreater(vpi_high, 0.0)

    def test_select_logic(self):
        root = DirichletNode(None, None, self.mcts.prior)
        child_a = DirichletNode(root, "A", [50, 0, 10])
        root.children["A"] = child_a
        child_b = DirichletNode(root, "B", [2, 1, 2])
        root.children["B"] = child_b

        selected = self.mcts._select(root)
        self.assertIn(selected, [child_a, child_b])

    def test_minimax_consolidation_uses_opponent_best_reply(self):
        parent_action = DirichletNode(None, "parent_action", [1.0, 0.1, 0.1])
        opponent_reply = DirichletNode(parent_action, "opponent_reply", [1.0, 0.1, 0.1])
        opponent_reply.force_certainty(-1.0)

        parent_action.children["opponent_reply"] = opponent_reply
        parent_action.is_expanded = True

        value = self.mcts._consolidate_minimax(parent_action)

        self.assertEqual(value, 1.0)
        self.assertEqual(parent_action.consolidated_value, 1.0)

    def test_final_choice_uses_consolidated_value_not_direct_mean(self):
        root = DirichletNode(None, None, self.mcts.prior)
        root.is_expanded = True

        forced_win = DirichletNode(root, "forced_win", [1.0, 0.1, 20.0])
        forced_win.is_expanded = True
        losing_reply = DirichletNode(forced_win, "only_reply", self.mcts.prior)
        losing_reply.force_certainty(-1.0)
        forced_win.children["only_reply"] = losing_reply

        tempting_leaf = DirichletNode(root, "tempting_leaf", [100.0, 0.1, 0.1])

        root.children["forced_win"] = forced_win
        root.children["tempting_leaf"] = tempting_leaf
        self.mcts.root = root

        direct_best = max(root.children.values(), key=lambda child: child.mu)
        consolidated_best = self.mcts._best_root_child()

        self.assertEqual(direct_best.action, "tempting_leaf")
        self.assertEqual(consolidated_best.action, "forced_win")
        self.assertEqual(forced_win.consolidated_value, 1.0)

    def test_search_returns_consolidated_root_choice(self):
        root = DirichletNode(None, None, self.mcts.prior)
        root.is_expanded = True

        forced_win = DirichletNode(root, "forced_win", [1.0, 0.1, 20.0])
        forced_win.is_expanded = True
        losing_reply = DirichletNode(forced_win, "only_reply", self.mcts.prior)
        losing_reply.force_certainty(-1.0)
        forced_win.children["only_reply"] = losing_reply

        tempting_leaf = DirichletNode(root, "tempting_leaf", [100.0, 0.1, 0.1])

        root.children["forced_win"] = forced_win
        root.children["tempting_leaf"] = tempting_leaf
        self.mcts.root = root

        root_state = MockGameState(name="root", legal_actions=["forced_win", "tempting_leaf"])
        action, sims = self.mcts.search(root_state, max_simulations=0, vpi_threshold=0.0)

        self.assertEqual(action, "forced_win")
        self.assertEqual(sims, 0)

    def test_solver_propagation_win_or_logic(self):
        """Test Immediate Win Propagation (OR Logic)."""
        # Root -> Child A
        root = DirichletNode(None, None, self.mcts.prior)
        child_a = DirichletNode(root, "A", self.mcts.prior)
        root.children["A"] = child_a

        # If Child A is a WIN (+1.0) for Root
        self.mcts._propagate_solve_status(child_a, 1.0)

        # Child A is a Win
        self.assertEqual(child_a.solved_outcome, 1.0)

        # Root found a winning move -> Root is Solved.
        # NOTE: propagate_solve_status calls parent with -outcome.
        # So Root should have called its parent with -1.0.
        # But here we just check the attribute on root isn't broken,
        # or check if propagate recurses correctly.
        # Since root has no parent, the recursion stops, but we verified
        # Child A called parent.

        # To test recursion, let's check a deeper tree
        # Gp -> Parent -> Child (+1)
        grandparent = DirichletNode(None, None, self.mcts.prior)
        parent = DirichletNode(grandparent, "P", self.mcts.prior)
        grandparent.children["P"] = parent
        child = DirichletNode(parent, "C", self.mcts.prior)
        parent.children["C"] = child

        # Child is a WIN (+1) for Parent
        self.mcts._propagate_solve_status(child, 1.0)

        self.assertEqual(child.solved_outcome, 1.0)
        # Parent found a win. Parent is effectively a LOSS (-1) for Grandparent.
        self.assertEqual(parent.solved_outcome, -1.0)

    def test_solver_propagation_loss_and_logic(self):
        """Test All-or-Nothing Loss Propagation (AND Logic)."""
        #
        grandparent = DirichletNode(None, None, self.mcts.prior)

        # Parent is an Action taken from Grandparent
        parent = DirichletNode(grandparent, "P", self.mcts.prior)
        grandparent.children["P"] = parent

        # Parent has 2 possible responses (C1, C2)
        parent.total_legal_moves = 2

        c1 = DirichletNode(parent, "C1", self.mcts.prior)
        c2 = DirichletNode(parent, "C2", self.mcts.prior)
        parent.children["C1"] = c1
        parent.children["C2"] = c2

        # 1. Child 1 is proven Loss (-1.0)
        self.mcts._propagate_solve_status(c1, -1.0)
        self.assertIsNone(parent.solved_outcome, "Parent waits for all children")

        # 2. Child 2 is proven Loss (-1.0)
        # Now Parent sees ALL children are Losses.
        # Parent Node (Action P) becomes a WIN (+1.0) for the Grandparent player.
        # (Because Grandparent played P, and P forces the opponent to lose).

        # Propagate recurses: Parent(+1.0) -> Grandparent(-1.0).
        self.mcts._propagate_solve_status(c2, -1.0)

        # Grandparent sees that the move 'P' leads to a Win for the opponent.
        # Therefore, the state at Grandparent is a Loss (assuming P was the only move tested so far).
        self.assertEqual(grandparent.solved_outcome, -1.0)

    def test_solver_disabled(self):
        mcts_dumb = BayesianMCTS(enable_solver=False)

        root = DirichletNode(None, None, mcts_dumb.prior)
        terminal_win = MockGameState(name="term", is_terminal=True, reward=1.0)
        root_state = MockGameState(name="root", legal_actions=[terminal_win])

        mcts_dumb._expand(root, root_state)
        child_node = root.children[terminal_win]

        # Action Value Check: Counts should represent Win (Index 0)
        self.assertGreater(child_node.counts[0], 100.0)
        self.assertIsNone(child_node.solved_outcome)


if __name__ == "__main__":
    unittest.main()
