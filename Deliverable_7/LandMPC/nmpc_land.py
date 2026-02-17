# LandMPC/nmpc_land.py

import numpy as np
import casadi as ca
from typing import Tuple
from scipy.signal import cont2discrete
from control import dlqr


class NmpcCtrl:

    def __init__(self, rocket, H: float, xs: np.ndarray, us: np.ndarray) -> None:

        self.f = lambda x, u: rocket.f_symbolic(x, u)[0]

        self.rocket = rocket
        self.Ts = float(rocket.Ts)
        self.H = float(H)
        self.N = int(round(self.H / self.Ts))

        self.xs = np.array(xs, dtype=float).reshape(12)
        self.us = np.array(us, dtype=float).reshape(4)

        self.xref = self.xs.copy()
        self.uref = self.us.copy()

        self._eps_u = 1e-6
        self._eps_x = 1e-6

        PAVG_MIN_MPC = 45.0

        self.u_lb = np.array(
            [
                -np.deg2rad(15.0) + self._eps_u,  # d1
                -np.deg2rad(15.0) + self._eps_u,  # d2
                PAVG_MIN_MPC + self._eps_u,       # Pavg
                -20.0 + self._eps_u,              # Pdiff
            ],
            dtype=float,
        )
        self.u_ub = np.array(
            [
                np.deg2rad(15.0) - self._eps_u,
                np.deg2rad(15.0) - self._eps_u,
                80.0 - self._eps_u,
                20.0 - self._eps_u,
            ],
            dtype=float,
        )

        self.alpha_max = np.deg2rad(9.8)
        self.beta_max = np.deg2rad(9.8)

        A_c, B_c = rocket.linearize(self.xs, self.us)
        Ad, Bd, _, _, _ = cont2discrete((A_c, B_c, np.zeros((1, 12)), np.zeros((1, 4))), self.Ts)
        Ad = np.array(Ad, dtype=float)
        Bd = np.array(Bd, dtype=float)

        Q_term = np.diag(
            [
                1.0, 1.0, 1.0,          # w
                250.0, 250.0, 600.0,    # angles
                30.0, 30.0, 40.0,       # v
                1200.0, 1200.0, 1800.0  # position
            ]
        )
        R_term = np.diag([10.0, 10.0, 0.5, 2.0])
        K_lqr, P_lqr, _ = dlqr(Ad, Bd, Q_term, R_term)
        self.K_lqr = np.array(K_lqr, dtype=float)
        self.P_lqr = np.array(P_lqr, dtype=float)

        # NMPC weights
        self.Q = np.diag(
            [
                2.0, 2.0, 2.0,            # w
                300.0, 300.0, 800.0,      # angles
                25.0, 25.0, 120.0,        # v
                1400.0, 1400.0, 2200.0    # positions
            ]
        )
        self.R = np.diag([25.0, 25.0, 0.4, 2.0])         # input magnitude penalty
        self.Rd = np.diag([40.0, 40.0, 0.08, 0.8])       # input rate penalty

        self._u_prev = self.uref.copy()

        self._prev_X = None
        self._prev_U = None

        self._setup_controller()

    def _rk4(self, x: ca.MX, u: ca.MX) -> ca.MX:
        dt = self.Ts
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5 * dt * k1, u)
        k3 = self.f(x + 0.5 * dt * k2, u)
        k4 = self.f(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _setup_controller(self) -> None:
        opti = ca.Opti()

        nx = 12
        nu = 4
        N = self.N

        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)

        x0_p = opti.parameter(nx)
        xref_p = opti.parameter(nx)
        u_prev_p = opti.parameter(nu)

        opti.subject_to(X[:, 0] == x0_p)
        for k in range(N):
            opti.subject_to(X[:, k + 1] == self._rk4(X[:, k], U[:, k]))

        for k in range(N):
            opti.subject_to(opti.bounded(self.u_lb[0], U[0, k], self.u_ub[0]))
            opti.subject_to(opti.bounded(self.u_lb[1], U[1, k], self.u_ub[1]))
            opti.subject_to(opti.bounded(self.u_lb[2], U[2, k], self.u_ub[2]))
            opti.subject_to(opti.bounded(self.u_lb[3], U[3, k], self.u_ub[3]))

            opti.subject_to(opti.bounded(-self.alpha_max, X[3, k], self.alpha_max))
            opti.subject_to(opti.bounded(-self.beta_max, X[4, k], self.beta_max))
            opti.subject_to(X[11, k] >= 0.0 + self._eps_x)

        opti.subject_to(opti.bounded(-self.alpha_max, X[3, N], self.alpha_max))
        opti.subject_to(opti.bounded(-self.beta_max, X[4, N], self.beta_max))
        opti.subject_to(X[11, N] >= 0.0 + self._eps_x)

        obj = 0
        uref = self.uref.reshape(nu)

        for k in range(N):
            e = X[:, k] - xref_p
            du = U[:, k] - uref
            obj += ca.mtimes([e.T, self.Q, e]) + ca.mtimes([du.T, self.R, du])

            if k == 0:
                dU = U[:, 0] - u_prev_p
            else:
                dU = U[:, k] - U[:, k - 1]
            obj += ca.mtimes([dU.T, self.Rd, dU])

        eN = X[:, N] - xref_p
        obj += ca.mtimes([eN.T, self.P_lqr, eN])

        opti.minimize(obj)

        opts = {
            "expand": True,
            "ipopt": {
                "print_level": 0,
                "max_iter": 300,
                "tol": 1e-4,
                "acceptable_tol": 1e-3,
                "acceptable_obj_change_tol": 1e-4,
                "sb": "yes",
                "linear_solver": "mumps",
                "warm_start_init_point": "yes",
            },
            "print_time": False,
        }
        opti.solver("ipopt", opts)

        self.opti = opti
        self.X = X
        self.U = U
        self.x0_p = x0_p
        self.xref_p = xref_p
        self.u_prev_p = u_prev_p

    def _clip_u(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(4)
        return np.minimum(np.maximum(u, self.u_lb), self.u_ub)

    def get_u(self, t0: float, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x0 = np.asarray(x0, dtype=float).reshape(12)

        self.opti.set_value(self.x0_p, x0)
        self.opti.set_value(self.xref_p, self.xref)
        self.opti.set_value(self.u_prev_p, self._u_prev)

        if self._prev_X is not None and self._prev_U is not None:
            Xg = np.hstack([self._prev_X[:, 1:], self._prev_X[:, -1:]])
            Ug = np.hstack([self._prev_U[:, 1:], self._prev_U[:, -1:]])
            self.opti.set_initial(self.X, Xg)
            self.opti.set_initial(self.U, Ug)
        else:
            Xg = np.tile(x0.reshape(12, 1), (1, self.N + 1))
            for k in range(self.N + 1):
                a = k / max(1, self.N)
                Xg[5, k] = (1 - a) * x0[5] + a * self.xref[5]      # gamma
                Xg[9, k] = (1 - a) * x0[9] + a * self.xref[9]      # x
                Xg[10, k] = (1 - a) * x0[10] + a * self.xref[10]   # y
                Xg[11, k] = (1 - a) * x0[11] + a * self.xref[11]   # z

            Ug = np.tile(self._u_prev.reshape(4, 1), (1, self.N))
            self.opti.set_initial(self.X, Xg)
            self.opti.set_initial(self.U, Ug)

        try:
            sol = self.opti.solve()
            Xsol = np.array(sol.value(self.X), dtype=float)
            Usol = np.array(sol.value(self.U), dtype=float)

            u0 = self._clip_u(Usol[:, 0].copy())

            self._u_prev = u0.copy()

            self._prev_X = Xsol
            self._prev_U = Usol

            t_ol = (t0 + self.Ts * np.arange(self.N + 1)).astype(float)

            return u0, Xsol, Usol, t_ol

        except Exception:
            u_safe = self._u_prev.copy()
            u_safe[0] = 0.0
            u_safe[1] = 0.0
            u_safe[2] = max(u_safe[2], self.us[2])
            u_safe[3] = 0.0
            u0 = self._clip_u(u_safe)

            self._u_prev = u0.copy()

            Xsol = np.tile(x0.reshape(12, 1), (1, self.N + 1))
            Usol = np.tile(u0.reshape(4, 1), (1, self.N))
            t_ol = (t0 + self.Ts * np.arange(self.N + 1)).astype(float)

            return u0, Xsol, Usol, t_ol
