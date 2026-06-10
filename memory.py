import random
import time
import math
from typing import List, Union, Dict, Any, Callable, Optional
from dataclasses import dataclass


@dataclass
class MemoryTrace:
    outcome: float
    timestamp: float
    encoding_strength: float


class EpisodicMemoryStore:
    def __init__(self, key_function: Optional[Callable[[Any], str]] = None):
        """
        Args:
            key_function: A function that takes a state object and returns a
                          unique, hashable key (usually a string).
                          If None, defaults to str(state).
        """
        self._store: Dict[str, List[MemoryTrace]] = {}
        self._key_function = key_function if key_function else str

    def _get_key(self, state: Any) -> str:
        return self._key_function(state)

    def add(self,
            state: Any,
            outcome: Union[float, List[float]],
            criticality: float = 1.0,
            timestamp: float = None):
        """
        Stores an experience with metadata for behavioral modeling.

        Args:
            state: The board state.
            outcome: The result (or list of results).
            criticality: Encoding strength multiplier (Default 1.0).
                         Use 2.0+ for 'Critical' positions to model flashbulb memories.
            timestamp: Real-world time of encoding. Defaults to time.time().
        """
        key = self._get_key(state)
        if timestamp is None:
            timestamp = time.time()

        if key not in self._store:
            self._store[key] = []

        # Normalize input to list
        outcomes = outcome if isinstance(outcome, list) else [outcome]

        for out in outcomes:
            trace = MemoryTrace(
                outcome=out,
                timestamp=timestamp,
                encoding_strength=criticality
            )
            self._store[key].append(trace)

    def retrieve(self,
                 state: Any,
                 current_time: float = None,
                 decay_rate: float = 0.0) -> List[float]:
        """
        Retrieves outcomes, processing them through decay and criticality logic.

        Args:
            state: The board state to lookup.
            current_time: The time of retrieval (defaults to time.time()).
            decay_rate: The exponent 'd' in the power law of forgetting (t^-d).
                        0.0 = No forgetting.

        Returns:
            A list of outcomes where the *quantity* of samples reflects
            the memory's current activation strength.
        """
        key = self._get_key(state)
        traces = self._store.get(key, [])

        if not traces:
            return []

        if current_time is None:
            current_time = time.time()

        retrieved_samples = []

        for trace in traces:
            # 1. Calculate Time Elapsed
            # Add small epsilon (1.0) to prevent division by zero or infinite spikes at t=0
            time_elapsed = max(0.0, current_time - trace.timestamp) + 1.0

            # 2. Calculate Activation (Power Law of Forgetting)
            # Base Activation = Strength * (Time^-d)
            decay_factor = math.pow(time_elapsed, -decay_rate)
            activation = trace.encoding_strength * decay_factor

            # 3. Probabilistic Reconstitution (Float -> Discrete Samples)
            # If activation is 2.5:
            # - We definitely add the outcome 2 times.
            # - We add the outcome a 3rd time with 50% probability.

            certain_count = int(activation)
            remainder = activation - certain_count

            # Add the certain copies
            retrieved_samples.extend([trace.outcome] * certain_count)

            # Probabilistically add the remainder
            if remainder > 0 and random.random() < remainder:
                retrieved_samples.append(trace.outcome)

        return retrieved_samples

    def forget(self, fraction: float):
        """
        Structural Forgetting (Random Neuron Loss).
        Deletes a random fraction (0.0-1.0) of trace objects.
        Distinct from 'decay', which weakens traces without deleting them.
        """
        if fraction <= 0.0: return
        if fraction >= 1.0:
            self._store.clear()
            return

        keys = list(self._store.keys())
        for key in keys:
            traces = self._store[key]
            current_count = len(traces)
            retain_count = int(current_count * (1.0 - fraction))

            if retain_count == 0:
                del self._store[key]
            elif retain_count < current_count:
                self._store[key] = random.sample(traces, retain_count)

    def get_stats(self) -> dict:
        """Metadata for analysis."""
        total_positions = len(self._store)
        total_traces = sum(len(v) for v in self._store.values())

        avg = total_traces / total_positions if total_positions > 0 else 0

        return {
            "unique_positions": total_positions,
            "total_traces": total_traces,
            "avg_traces_per_pos": avg
        }
