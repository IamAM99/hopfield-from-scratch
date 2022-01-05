"""Hopfiled network implementation.
"""

import numpy as np
from utils import sgn


class Hopfield:
    """A class to make a Hopfield network.
    """

    def __init__(self, P: np.ndarray):
        """Calculates the weight matrix based on the input patterns 'P'.
        """
        self.W = (P @ P.T) - (P.shape[1] * np.eye(P.shape[0]))

    def predict(self, v: np.ndarray, update_type: str = "sync", iterations: int = 100):
        """Predict which saved pattern in the network the given pattern is closest to.
        """
        # make a history of size '5', to check for limit cycles in the 'sync' update_type
        hist_size = 5
        v_hist = [*([None] * (hist_size - 1)), v]

        # track whether break was called or not while using 'sync' update_type
        break_called = False

        for _ in range(iterations):
            if update_type == "sync":
                v_new = sgn(self.W @ v_hist[-1])

                if np.all(v_new == v_hist[-1]):
                    print(
                        "=" * 67,
                        "Pattern is no longer improving. Returning the last updated pattern.",
                        "=" * 67,
                        sep="\n",
                        end="\n\n",
                    )
                    break_called = True
                    break
                elif np.any([np.all(hist == v_new) for hist in v_hist[:-1]]):
                    print(
                        "=" * 58,
                        "Reached a limit cycle. Returning the last updated pattern.",
                        "=" * 58,
                        sep="\n",
                        end="\n\n",
                    )
                    break_called = True
                    break
                else:
                    v_hist = [*(v_hist[1:]), v_new]

            elif update_type == "async":
                v_new = np.copy(v_hist[-1])
                for _ in range(v.shape[0]):
                    idx = np.random.choice(range(v_hist[-1].shape[0]))
                    updated = sgn(self.W[idx].reshape((1, -1)) @ v_hist[-1])
                    if v_new[idx] != updated:
                        v_new[idx] = updated.item()
                        break
                v_hist = [*(v_hist[1:]), v_new]

        if break_called is False:
            print(
                "=" * 73,
                "Maximum number of iterations reached. Returning the last updated pattern.",
                "=" * 73,
                sep="\n",
                end="\n\n",
            )

        return np.asarray(v_new, dtype=int)
