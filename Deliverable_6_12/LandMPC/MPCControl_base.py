import cvxpy as cp
import numpy as np
from control import dlqr
from scipy.signal import cont2discrete


class MPCControl_base:

    x_ids: np.ndarray
    u_ids: np.ndarray

    def __init__(self, A, B, xs, us, Ts, H) -> None:
        self.Ts = float(Ts)
        self.H = float(H)
        self.N = int(round(self.H / self.Ts))

        self.nx = int(self.x_ids.shape[0])
        self.nu = int(self.u_ids.shape[0])

        A_red = A[np.ix_(self.x_ids, self.x_ids)]
        B_red = B[np.ix_(self.x_ids, self.u_ids)]

        self.A, self.B = self._discretize(A_red, B_red, self.Ts)

        self.xs = np.asarray(xs[self.x_ids], dtype=float).copy()
        self.us = np.asarray(us[self.u_ids], dtype=float).copy()

        self.last_status = None
        self.debug = False

        self._prev_x = None
        self._prev_u = None

        self._setup_controller()

    def _setup_controller(self) -> None:
        raise NotImplementedError

    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        nx, nu = B.shape
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        Ad, Bd, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
        return np.array(Ad, dtype=float), np.array(Bd, dtype=float)
    
    @staticmethod
    def _clip_strict(v: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        v = np.array(v, dtype=float, copy=True)
        lb = np.array(lb, dtype=float, copy=False)
        ub = np.array(ub, dtype=float, copy=False)

        if np.any(np.isfinite(lb)):
            lb_in = np.where(np.isfinite(lb), np.nextafter(lb, np.inf), lb)
            v = np.maximum(v, lb_in)
        if np.any(np.isfinite(ub)):
            ub_in = np.where(np.isfinite(ub), np.nextafter(ub, -np.inf), ub)
            v = np.minimum(v, ub_in)
        return v

    def _setup_mpc_problem(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        x_lb: np.ndarray,
        x_ub: np.ndarray,
        u_lb: np.ndarray,
        u_ub: np.ndarray,
        *,
        soft_state: bool = False,
        soft_weight: float = 2e4,
        use_terminal_set: bool = True,
    ) -> None:

        eps_x = 5e-4
        eps_u = 5e-4
        x_lb = np.where(np.isfinite(x_lb), x_lb + eps_x, x_lb)
        x_ub = np.where(np.isfinite(x_ub), x_ub - eps_x, x_ub)
        u_lb = np.where(np.isfinite(u_lb), u_lb + eps_u, u_lb)
        u_ub = np.where(np.isfinite(u_ub), u_ub - eps_u, u_ub)

        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)

        self.x_lb = np.array(x_lb, dtype=float)
        self.x_ub = np.array(x_ub, dtype=float)
        self.u_lb = np.array(u_lb, dtype=float)
        self.u_ub = np.array(u_ub, dtype=float)

        K_lqr, P_lqr, _ = dlqr(self.A, self.B, self.Q, self.R)
        self.K = np.array(K_lqr, dtype=float)
        self.P = np.array(P_lqr, dtype=float)

        self._x = cp.Variable((self.nx, self.N + 1))
        self._u = cp.Variable((self.nu, self.N))

        self._s = None
        if soft_state:
            self._s = cp.Variable((self.nx, self.N + 1), nonneg=True)

        self._x0_p = cp.Parameter(self.nx)
        self._xref_p = cp.Parameter(self.nx)
        self._uref_p = cp.Parameter(self.nu)

        cons = [self._x[:, 0] == self._x0_p]

        for k in range(self.N):
            cons += [
                self._x[:, k + 1]
                == self.A @ (self._x[:, k] - self.xs)
                + self.B @ (self._u[:, k] - self.us)
                + self.xs
            ]

            for i in range(self.nu):
                if np.isfinite(self.u_lb[i]):
                    cons += [self._u[i, k] >= self.u_lb[i]]
                if np.isfinite(self.u_ub[i]):
                    cons += [self._u[i, k] <= self.u_ub[i]]

            for i in range(self.nx):
                if np.isfinite(self.x_lb[i]):
                    if self._s is None:
                        cons += [self._x[i, k] >= self.x_lb[i]]
                    else:
                        cons += [self._x[i, k] >= self.x_lb[i] - self._s[i, k]]
                if np.isfinite(self.x_ub[i]):
                    if self._s is None:
                        cons += [self._x[i, k] <= self.x_ub[i]]
                    else:
                        cons += [self._x[i, k] <= self.x_ub[i] + self._s[i, k]]

        if use_terminal_set:
            kN = self.N
            for i in range(self.nx):
                if np.isfinite(self.x_lb[i]):
                    if self._s is None:
                        cons += [self._x[i, kN] >= self.x_lb[i]]
                    else:
                        cons += [self._x[i, kN] >= self.x_lb[i] - self._s[i, kN]]
                if np.isfinite(self.x_ub[i]):
                    if self._s is None:
                        cons += [self._x[i, kN] <= self.x_ub[i]]
                    else:
                        cons += [self._x[i, kN] <= self.x_ub[i] + self._s[i, kN]]

        # Objective
        obj = 0
        for k in range(self.N):
            e = self._x[:, k] - self._xref_p
            du = self._u[:, k] - self._uref_p
            obj += cp.quad_form(e, self.Q) + cp.quad_form(du, self.R)
            if self._s is not None:
                obj += soft_weight * cp.sum(self._s[:, k])

        eN = self._x[:, self.N] - self._xref_p
        obj += cp.quad_form(eN, self.P)
        if self._s is not None:
            obj += soft_weight * cp.sum(self._s[:, self.N])

        self.ocp = cp.Problem(cp.Minimize(obj), cons)

    def _solve_ocp(self) -> bool:

        solve_kwargs = dict(warm_start=True, verbose=False)

        if "OSQP" in cp.installed_solvers():
            solve_kwargs.update(
                dict(
                    solver=cp.OSQP,
                    eps_abs=1e-5,
                    eps_rel=1e-5,
                    max_iter=200000,
                    polish=True,
                )
            )
        elif "GUROBI" in cp.installed_solvers():
            solve_kwargs.update(dict(solver=cp.GUROBI))
        elif "CLARABEL" in cp.installed_solvers():
            solve_kwargs.update(dict(solver=cp.CLARABEL))
        elif "ECOS" in cp.installed_solvers():
            solve_kwargs.update(dict(solver=cp.ECOS))

        self.ocp.solve(**solve_kwargs)
        self.last_status = self.ocp.status
        working = self.ocp.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        if self.debug:
            stats = getattr(self.ocp, "solver_stats", None)
            iters = getattr(stats, "num_iters", None) if stats is not None else None
            print(f"[{self.__class__.__name__}] status={self.ocp.status} iters={iters} obj={self.ocp.value}")

        return working

    def get_u(self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None):
        if x_target is None:
            x_target = self.xs.copy()
        if u_target is None:
            u_target = self.us.copy()

        x0 = np.asarray(x0, dtype=float).reshape(self.nx)
        x_target = np.asarray(x_target, dtype=float).reshape(self.nx)
        u_target = np.asarray(u_target, dtype=float).reshape(self.nu)

        self._x0_p.value = x0
        self._xref_p.value = x_target
        self._uref_p.value = u_target

        if (self._prev_x is not None) and (self._prev_u is not None):
            x_ws = np.hstack([self._prev_x[:, 1:], self._prev_x[:, -1:]])
            u_ws = np.hstack([self._prev_u[:, 1:], self._prev_u[:, -1:]])
            self._x.value = x_ws
            self._u.value = u_ws

        working = self._solve_ocp()

        if not working:
            if self._prev_u is not None:
                u0 = self._prev_u[:, 0].copy()
            else:
                u0 = self.us.copy()

            u0 = self._clip_strict(u0, self.u_lb, self.u_ub)

            x_traj = np.zeros((self.nx, self.N + 1), dtype=float)
            u_traj = np.zeros((self.nu, self.N), dtype=float)
            x_traj[:, 0] = x0
            for k in range(self.N):
                u_traj[:, k] = u0
                x_traj[:, k + 1] = (
                    self.A @ (x_traj[:, k] - self.xs)
                    + self.B @ (u0 - self.us)
                    + self.xs
                )
            return u0, x_traj, u_traj

        x_traj = np.array(self._x.value, dtype=float)
        u_traj = np.array(self._u.value, dtype=float)

        u_traj = self._clip_strict(
            u_traj,
            self.u_lb.reshape(-1, 1),
            self.u_ub.reshape(-1, 1),
        )

        x_roll = np.zeros((self.nx, self.N + 1), dtype=float)
        x_roll[:, 0] = x0
        for k in range(self.N):
            x_roll[:, k + 1] = (
                self.A @ (x_roll[:, k] - self.xs)
                + self.B @ (u_traj[:, k] - self.us)
                + self.xs
            )

        self._prev_x = x_roll
        self._prev_u = u_traj

        u0 = u_traj[:, 0].copy()
        return u0, x_roll, u_traj
