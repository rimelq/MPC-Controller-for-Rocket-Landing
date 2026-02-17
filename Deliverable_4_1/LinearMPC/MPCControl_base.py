import cvxpy as cp
import numpy as np
from control import dlqr
from scipy.signal import cont2discrete
from mpt4py import Polyhedron 


class MPCControl_base:
    """Complete states indices"""
    x_ids: np.ndarray
    u_ids: np.ndarray

    """Optimization system"""
    A: np.ndarray
    B: np.ndarray
    xs: np.ndarray
    us: np.ndarray
    nx: int
    nu: int
    Ts: float
    H: float
    N: int

    """LQR"""
    Q: np.ndarray
    R: np.ndarray
    K: np.ndarray
    P: np.ndarray
    Acl: np.ndarray
    Xf: object
    Af: np.ndarray
    bf: np.ndarray

    """Bounds"""
    x_lb: np.ndarray
    x_ub: np.ndarray
    u_lb: np.ndarray
    u_ub: np.ndarray

    """Optimization problem"""
    ocp: cp.Problem

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        xs: np.ndarray,
        us: np.ndarray,
        Ts: float,
        H: float,
    ) -> None:
        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)
        self.nx = int(self.x_ids.shape[0])
        self.nu = int(self.u_ids.shape[0])

        A_red = A[np.ix_(self.x_ids, self.x_ids)]
        B_red = B[np.ix_(self.x_ids, self.u_ids)]

        self.A, self.B = self._discretize(A_red, B_red, Ts)

        self.xs = xs[self.x_ids].astype(float).copy()
        self.us = us[self.u_ids].astype(float).copy()

        self._prev_x = None
        self._prev_u = None

        self.last_status = None
        self.debug = False

        self._setup_controller()

    def _setup_controller(self) -> None:
        raise NotImplementedError

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
           

        solved = False
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
        solved = self.ocp.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        if self.debug:
            stats = getattr(self.ocp, "solver_stats", None)
            iters = getattr(stats, "num_iters", None) if stats is not None else None
            print(
                f"[{self.__class__.__name__}] status={self.ocp.status} iters={iters} obj={self.ocp.value}"
            )


        if not solved:
            if self._prev_u is not None:
                u0 = self._prev_u[:, 0].copy()
            else:
                u0 = self.us.copy()

            u0 = self._clip_strict(u0, self.u_lb, self.u_ub)

            x_traj = np.zeros((self.nx, self.N + 1))
            u_traj = np.zeros((self.nu, self.N))
            x_traj[:, 0] = x0
            for k in range(self.N):
                u_traj[:, k] = u0
                x_traj[:, k + 1] = (
                    self.A @ (x_traj[:, k] - self.xs)
                    + self.B @ (u0 - self.us)
                    + self.xs
                )

            return u0, x_traj, u_traj

        u_traj = np.array(self._u.value, dtype=float)
        u_traj = self._clip_strict(
            u_traj, self.u_lb.reshape(-1, 1), self.u_ub.reshape(-1, 1)
        )

        x_traj = np.zeros((self.nx, self.N + 1), dtype=float)
        x_traj[:, 0] = x0
        for k in range(self.N):
            x_traj[:, k + 1] = (
                self.A @ (x_traj[:, k] - self.xs)
                + self.B @ (u_traj[:, k] - self.us)
                + self.xs
            )

        u0 = u_traj[:, 0].copy()

        self._prev_x = x_traj
        self._prev_u = u_traj

        return u0, x_traj, u_traj

    def _setup_mpc_problem(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        x_lb: np.ndarray,
        x_ub: np.ndarray,
        u_lb: np.ndarray,
        u_ub: np.ndarray,
        terminal_maxiter: int = 200,
        use_terminal_set: bool = False,
        soft_state: bool = True,
        rho_x: float = 1e6,
    ) -> None:
        
        eps_u = 5e-4
        u_lb = np.where(np.isfinite(u_lb), u_lb + eps_u, u_lb)
        u_ub = np.where(np.isfinite(u_ub), u_ub - eps_u, u_ub)

        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)

        self.x_lb = np.array(x_lb, dtype=float)
        self.x_ub = np.array(x_ub, dtype=float)
        self.u_lb = np.array(u_lb, dtype=float)
        self.u_ub = np.array(u_ub, dtype=float)

        K, P, _ = dlqr(self.A, self.B, self.Q, self.R)
        self.K = np.array(K, dtype=float)
        self.P = np.array(P, dtype=float)
        self.Acl = self.A - self.B @ self.K

        self.Af, self.bf, self.Xf = None, None, None
        if use_terminal_set:
            self.Af, self.bf, self.Xf = self._compute_terminal_invariant_set(
                maxiter=terminal_maxiter
            )

        self._x = cp.Variable((self.nx, self.N + 1))
        self._u = cp.Variable((self.nu, self.N))

        self._sx = None
        if soft_state:
            self._sx = cp.Variable((self.nx, self.N + 1), nonneg=True)

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
                lb = self.x_lb[i]
                ub = self.x_ub[i]
                if np.isfinite(lb):
                    if self._sx is None:
                        cons += [self._x[i, k] >= lb]
                    else:
                        cons += [self._x[i, k] >= lb - self._sx[i, k]]
                if np.isfinite(ub):
                    if self._sx is None:
                        cons += [self._x[i, k] <= ub]
                    else:
                        cons += [self._x[i, k] <= ub + self._sx[i, k]]

        for i in range(self.nx):
            lb = self.x_lb[i]
            ub = self.x_ub[i]
            if np.isfinite(lb):
                if self._sx is None:
                    cons += [self._x[i, self.N] >= lb]
                else:
                    cons += [self._x[i, self.N] >= lb - self._sx[i, self.N]]
            if np.isfinite(ub):
                if self._sx is None:
                    cons += [self._x[i, self.N] <= ub]
                else:
                    cons += [self._x[i, self.N] <= ub + self._sx[i, self.N]]

        if (self.Af is not None) and (self.bf is not None):
            cons += [self.Af @ (self._x[:, self.N] - self._xref_p) <= self.bf]

        obj = 0
        for k in range(self.N):
            e = self._x[:, k] - self._xref_p
            du = self._u[:, k] - self._uref_p
            obj += cp.quad_form(e, self.Q) + cp.quad_form(du, self.R)

        eN = self._x[:, self.N] - self._xref_p
        obj += cp.quad_form(eN, self.P)

        if self._sx is not None:
            obj += float(rho_x) * cp.sum(self._sx)

        self.ocp = cp.Problem(cp.Minimize(obj), cons)

    def _compute_terminal_invariant_set(
        self, maxiter: int = 200
    ) -> tuple[np.ndarray, np.ndarray, object]:
        if Polyhedron is None:
            eps = 1e-3
            Af = np.vstack([np.eye(self.nx), -np.eye(self.nx)])
            bf = eps * np.ones(2 * self.nx)
            return Af, bf, None

        A_rows = []
        b_rows = []

        for i in range(self.nx):
            if np.isfinite(self.x_ub[i]):
                a = np.zeros(self.nx)
                a[i] = 1.0
                A_rows.append(a)
                b_rows.append(self.x_ub[i] - self.xs[i])
            if np.isfinite(self.x_lb[i]):
                a = np.zeros(self.nx)
                a[i] = -1.0
                A_rows.append(a)
                b_rows.append(-(self.x_lb[i] - self.xs[i]))

        A_rows.append((-self.K).reshape(self.nu, self.nx))
        b_rows.append((self.u_ub - self.us).reshape(self.nu))
        A_rows.append((self.K).reshape(self.nu, self.nx))
        b_rows.append((self.us - self.u_lb).reshape(self.nu))

        A0 = np.vstack([Ai if Ai.ndim == 2 else Ai.reshape(1, -1) for Ai in A_rows])
        b0 = np.hstack(
            [bi if np.ndim(bi) > 0 else np.array([bi]) for bi in b_rows]
        ).astype(float)

        X = Polyhedron.from_Hrep(A=A0, b=b0)
        X.minHrep(inplace=True)

        for _ in range(maxiter):
            Apre = X.A @ self.Acl
            bpre = X.b

            Anew = np.vstack([A0, Apre])
            bnew = np.hstack([b0, bpre])

            Xnew = Polyhedron.from_Hrep(A=Anew, b=bnew)
            Xnew.minHrep(inplace=True)

            if (Xnew.A.shape[0] == X.A.shape[0]) and np.allclose(
                np.sort(Xnew.b), np.sort(X.b), atol=1e-9, rtol=1e-9
            ):
                X = Xnew
                break

            X = Xnew

        return np.array(X.A, dtype=float), np.array(X.b, dtype=float), X

    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float) -> tuple[np.ndarray, np.ndarray]:
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
