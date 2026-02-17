import numpy as np
from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    # [vz], input [Pavg]
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    def _setup_controller(self) -> None:
        
        Q = np.diag([50.0])
        R = np.diag([0.3])

        x_lb = np.array([-np.inf])
        x_ub = np.array([+np.inf])

        u_lb = np.array([50.0])
        u_ub = np.array([80.0])

        self._setup_mpc_problem(
            Q, R, x_lb, x_ub, u_lb, u_ub,
            terminal_maxiter=0,
            use_terminal_set=False,
            soft_state=False
        )
