import numpy as np
from .MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

    def _setup_controller(self) -> None:
        Q = np.diag([100, 5000]) 
        R = np.diag([20.0])

        x_lb = np.array([-np.inf, -np.inf])
        x_ub = np.array([+np.inf, +np.inf])

        u_lb = np.array([-20.0])
        u_ub = np.array([+20.0])

        self._setup_mpc_problem(
            Q, R, x_lb, x_ub, u_lb, u_ub,
            terminal_maxiter=0,
            use_terminal_set=False,
            soft_state=False
        )
