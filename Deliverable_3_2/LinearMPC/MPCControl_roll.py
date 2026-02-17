import numpy as np
from .MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

    def _setup_controller(self) -> None:

        Q = np.diag([80, 200.0])   
        R = np.diag([1])

        # Constraints
        x_lb = np.array([-np.inf, -np.inf])
        x_ub = np.array([+np.inf, +np.inf])
        u_lb = np.array([-20.0])
        u_ub = np.array([+20.0])

        self._setup_mpc_problem(Q, R, x_lb, x_ub, u_lb, u_ub, terminal_maxiter=250)

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().get_u(x0, x_target, u_target)

