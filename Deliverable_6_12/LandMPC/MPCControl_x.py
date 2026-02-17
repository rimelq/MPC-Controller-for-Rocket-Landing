import numpy as np
from .MPCControl_base import MPCControl_base


class MPCControl_x(MPCControl_base):

    x_ids: np.ndarray = np.array([1, 4, 6, 9])
    u_ids: np.ndarray = np.array([1])

    def _setup_controller(self) -> None:
        beta_max = np.deg2rad(5.0) 
        d2_max = np.deg2rad(15.0)

        Q = np.diag([10.0, 80.0, 6.0, 350.0]) 
        R = np.diag([8.0])  

        x_lb = np.array([-np.inf, -beta_max, -np.inf, -np.inf])
        x_ub = np.array([+np.inf, +beta_max, +np.inf, +np.inf])

        u_lb = np.array([-d2_max])
        u_ub = np.array([+d2_max])

        self._setup_mpc_problem(
            Q, R, x_lb, x_ub, u_lb, u_ub,
            soft_state=True,
            soft_weight=1e6,  
            use_terminal_set=False,
        )
