import numpy as np
from .MPCControl_base import MPCControl_base


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])

    def _setup_controller(self) -> None:
        alpha_max = np.deg2rad(10.0)

        Q = np.diag([1.0, 15.0, 0.2])
        R = np.diag([20.0])

        # Constrains
        x_lb = np.array([-np.inf, -alpha_max, -np.inf])
        x_ub = np.array([+np.inf, +alpha_max, +np.inf])
        d1_max = np.deg2rad(15.0)
        u_lb = np.array([-d1_max])
        u_ub = np.array([+d1_max])

        self._setup_mpc_problem(Q, R, x_lb, x_ub, u_lb, u_ub, terminal_maxiter=250)

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().get_u(x0, x_target, u_target)

