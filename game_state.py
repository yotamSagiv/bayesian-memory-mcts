# game_state.py
from abc import ABC, abstractmethod
from typing import List, Any
import bulletchess
import random


class AbstractGameState(ABC):
    """
    The interface the MCTS expects.
    Any game implementation (Chess, Go, Connect4) must adhere to this.
    """

    @abstractmethod
    def get_legal_actions(self) -> List[Any]: pass

    @abstractmethod
    def apply_action(self, action: Any) -> 'AbstractGameState': pass

    @abstractmethod
    def is_terminal(self) -> bool: pass

    @abstractmethod
    def get_reward(self) -> float: pass

    # --- Heuristics required for the rollout policy ---
    @abstractmethod
    def is_capture(self, action: Any) -> bool: pass

    @abstractmethod
    def gives_check(self, action: Any) -> bool: pass

    @abstractmethod
    def gives_mate(self, action: Any) -> bool: pass


class ChessGameState(AbstractGameState):
    def __init__(self, board: bulletchess.Board = None, fen: str = None):
        """
        Initialize with an existing bulletchess Board or a FEN string.
        """
        if board:
            self.board = board
        elif fen:
            self.board = bulletchess.Board.from_fen(fen)
        else:
            self.board = bulletchess.Board()

    def get_legal_actions(self) -> List[bulletchess.Move]:
        """Returns a list of legal moves from the current position."""
        return self.board.legal_moves()

    def apply_action(self, action: bulletchess.Move) -> 'ChessGameState':
        """
        Returns a new state with the action applied.
        Uses a fast C-level copy to ensure immutability of the previous state.
        """
        new_board = self.board.copy()
        new_board.apply(action)
        return ChessGameState(board=new_board)

    def is_terminal(self) -> bool:
        """
        Checks if the game is over.
        MATE covers both Checkmate and Stalemate.
        DRAW covers 50-move rule, repetition, etc.
        """
        return (self.board in bulletchess.MATE) or (self.board in bulletchess.DRAW)

    def get_reward(self) -> float:
        """
        Returns +1 (Win), 0 (Draw), -1 (Loss)
        from the perspective of the player who JUST moved (the previous turn).
        """
        # If the board is in CHECKMATE, the current player has lost.
        # Therefore, the player who just moved is the winner.
        if self.board in bulletchess.CHECKMATE:
            return 1.0

        # Check for Draw conditions (Stalemate, Repetition, Insufficient Material)
        if self.board in bulletchess.DRAW or self.board in bulletchess.STALEMATE:
            return 0.0

        return 0.0

    # --- Heuristic Implementations ---

    def is_capture(self, action: bulletchess.Move) -> bool:
        """Check if the destination square is currently occupied."""
        return self.board[action.destination] is not None

    def gives_check(self, action: bulletchess.Move) -> bool:
        """
        Simulates the move to see if it results in a check state.
        Must apply and undo to preserve board state.
        """
        self.board.apply(action)
        is_check = (self.board in bulletchess.CHECK)
        self.board.undo()
        return is_check

    def gives_mate(self, action: bulletchess.Move) -> bool:
        """
        Simulates the move to see if it results in immediate checkmate.
        This is expensive, so we prioritize cheaper checks in the rollout first.
        """
        self.board.apply(action)
        is_mate = (self.board in bulletchess.CHECKMATE)
        self.board.undo()
        return is_mate

    def __repr__(self):
        # Returns: "rnbqk... w KQkq - 0 1"
        full_fen = self.board.fen()

        # Split by space and take first 4 parts:
        # 1. Piece Placement
        # 2. Active Color
        # 3. Castling Rights
        # 4. En Passant Target
        # (Ignore 5. Halfmove Clock and 6. Fullmove Number)
        return " ".join(full_fen.split(" ")[:4])
