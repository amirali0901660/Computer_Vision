"""Check a small alpha-beta swap against exact exhaustive enumeration.

Run: python code/utils/check_graph_cuts.py
This is a limited correctness test, not proof of global graph-cut optimality.
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization import edge_weights, energy, graph_cuts


def run():
    """Test 10 random costs each with and without pixels fixed to a third label."""
    rng = np.random.default_rng(2026)
    candidates = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    weight = 0.7
    cases = {
        "all_active": np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32),
        "with_fixed_pixels": np.array([[0, 1, 2], [1, 0, 2]], dtype=np.int32),
    }

    for name, initial in cases.items():
        for trial in range(10):
            cost = rng.random((3, 2, 3)).astype(np.float32)
            image = rng.random((2, 3, 3)).astype(np.float32)
            horizontal, vertical = edge_weights(image)

            active = (initial == 0) | (initial == 1)
            positions = np.argwhere(active)
            best_energy = float("inf")

            # Try every possible assignment of labels 0/1 at active pixels.
            for combination in product((0, 1), repeat=len(positions)):
                labels = initial.copy()
                for (y, x), label in zip(positions, combination):
                    labels[y, x] = label
                score = energy(cost, labels, candidates, horizontal, vertical, weight)
                best_energy = min(best_energy, score)

            # The pair limit selects only (0,1) on this 3-label example.
            _, result = graph_cuts(cost, candidates, image, weight=weight,
                                   cycles=1, initial=initial, max_pairs=1)
            graph_energy = energy(cost, result, candidates,
                                  horizontal, vertical, weight)

            if not np.array_equal(result[~active], initial[~active]):
                raise AssertionError(f"{name}, trial {trial}: fixed labels changed")
            if abs(graph_energy - best_energy) > 1e-4:
                raise AssertionError(f"{name}, trial {trial}: expected {best_energy}, "
                                     f"got {graph_energy}")
        print(f"{name}: 10/10 passed")
    print("All 20 swap tests passed.")


if __name__ == "__main__":
    run()
