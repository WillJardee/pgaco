from abc import abstractmethod
import ast
import pickle
import warnings
import numpy as np
from pgaco.utils import post_init_decorator


class ACOBase():
    @post_init_decorator
    def __init__(self,
                 *,
                 seed: int | None = None,
                 save_file: str | None = None,
                 checkpoint_file: str | None = None,
                 checkpoint_res: int = 10,
                 **kwargs):
        for key in kwargs:
            warnings.warn(f"Parameter '{key}' is not recognized and will be ignored.",
                          stacklevel=2)

        self._savefile = save_file
        self._checkpointfile = checkpoint_file
        self._checkpoint_res = checkpoint_res
        self._seed = seed
        self._rng               =   np.random.default_rng(seed=self._seed)

        self._iteration         =   0

        self._heuristic_table = np.array([])

    def _post_init(self):
        self._validate_params()

    def _between(self, value, *, lower: int | None = None, upper: int | None = None, inclusive: bool = False) -> bool:
        assert isinstance(inclusive, bool)
        if lower is not None:
            if inclusive and value <= lower:
                return False
            elif (not inclusive) and value < lower:
                return False
        if upper is not None:
            if inclusive and value >= upper:
                return False
            elif (not inclusive) and value > upper:
                return False
        return True


    def _validate_params(self):
        assert isinstance(self._savefile, (str, type(None)))
        assert isinstance(self._checkpointfile, (str, type(None)))
        assert isinstance(self._checkpoint_res, int)
        assert self._between(self._checkpoint_res, lower = 0)

    def _save_params(self) -> None:
        pass

    def _checkpoint(self) -> None:
        """Pickles self to disk."""
        if self._checkpointfile is None:
            return
        with open(self._checkpointfile, "wb") as f:
            pickle.dump(self, f)

    def _save(self) -> None:
        """Save learned pheromone table to disk."""
        if self._savefile is None:
            return
        with open(self._savefile, "ab") as f:
            f.write(self._heuristic_table.astype(np.float64).tobytes())

    def _load(self, filename) -> tuple[dict, np.ndarray]:
        """Load learned pheromone table from disk."""
        with open(filename, "rb") as f:
            params = ast.literal_eval(f.readline().decode())
            tau_table = np.frombuffer(f.read(), dtype=np.float64)
            tau_table = tau_table.reshape((-1, params["size"], params["size"]))
        return params, tau_table

    def _restore(self) -> None:
        pass

    @abstractmethod
    def _gradient_update(self) -> None:
        pass

    @abstractmethod
    def take_step(self, step: int) -> tuple[float, np.ndarray]:
        pass

    @abstractmethod
    def run(self, max_iter: int) -> tuple[float, np.ndarray]:
        pass

