import numpy as np
from control import dlqr
from .MPCControl_base import MPCControl_base
import cvxpy as cp



class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    def _setup_controller(self) -> None:

        self.Q = np.diag([50.0])
        self.R = np.diag([0.3])

        self.x_lb = np.array([-np.inf])
        self.x_ub = np.array([+np.inf])

        self.u_lb = np.array([40.0])
        self.u_ub = np.array([80.0])

        K, P, _ = dlqr(self.A, self.B, self.Q, self.R)
        self.P = np.array(P, dtype=float)

        A_disc = float(self.A[0, 0])
        B_disc = float(self.B[0, 0])
        xs = float(self.xs[0])  
        us = float(self.us[0]) 

        self._A_disc = A_disc
        self._B_disc = B_disc
        self._xs = xs
        self._us = us

        self._c = (1.0 - A_disc) * xs - B_disc * us

        self._Aaug = np.array([[A_disc, B_disc],
                               [0.0, 1.0]], dtype=float)
        self._Baug = np.array([[B_disc],
                               [0.0]], dtype=float)
        self._caug = np.array([[self._c],
                               [0.0]], dtype=float)
        self._Caug = np.array([[1.0, 0.0]], dtype=float)

        Qobs = np.diag([1.0, 100.0])
        Robs = np.array([[0.1]], dtype=float)

        Kobs, _, _ = dlqr(self._Aaug.T, self._Caug.T, Qobs, Robs)
        self._L = np.array(Kobs.T, dtype=float)

        self._xhat = None

        self._log_dhat = []
        self._log_vzhat = []

        self._build_offset_free_mpc()

    def _build_offset_free_mpc(self):

        N = self.N
        nx, nu = self.nx, self.nu
        A, B = float(self.A[0, 0]), float(self.B[0, 0])

        self._mpc_x = cp.Variable((nx, N + 1))
        self._mpc_u = cp.Variable((nu, N))

        self._mpc_x0 = cp.Parameter(nx)         
        self._mpc_xs = cp.Parameter(nx)        
        self._mpc_us = cp.Parameter(nu)        
        self._mpc_d = cp.Parameter(1)          

        cons = [self._mpc_x[:, 0] == self._mpc_x0]

        for k in range(N):
            cons += [
                self._mpc_x[0, k+1] == A * (self._mpc_x[0, k] - self._mpc_xs[0])
                                     + B * (self._mpc_u[0, k] - self._mpc_us[0])
                                     + self._mpc_xs[0]
            ]

            cons += [self._mpc_u[0, k] >= float(self.u_lb[0])]
            cons += [self._mpc_u[0, k] <= float(self.u_ub[0])]

        Q, R, P = float(self.Q[0, 0]), float(self.R[0, 0]), float(self.P[0, 0])
        cost = 0
        for k in range(N):
            cost += Q * cp.square(self._mpc_x[0, k] - self._mpc_xs[0])
            cost += R * cp.square(self._mpc_u[0, k] - self._mpc_us[0])
        cost += P * cp.square(self._mpc_x[0, N] - self._mpc_xs[0]) 

        self._mpc_prob = cp.Problem(cp.Minimize(cost), cons)

    def reset_estimator(self) -> None:
        self._xhat = None
        self._u_prev = None
        self._log_dhat = []
        self._log_vzhat = []

    def get_estimator_logs(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self._log_dhat, dtype=float), np.array(self._log_vzhat, dtype=float)

    def _compute_target(self, d_hat: float, x_ref: float) -> tuple[float, float]:

        A = self._A_disc
        B = self._B_disc
        c = self._c

        xs_new = x_ref

        us_new = ((1.0 - A) * xs_new - B * d_hat - c) / B

        return xs_new, us_new

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ):
        if x_target is None:
            x_ref = 0.0
        else:
            x_ref = float(np.asarray(x_target).reshape(-1)[0])

        y_meas = float(np.asarray(x0).reshape(-1)[0])

        if self._xhat is None:
            self._xhat = np.array([y_meas, 0.0], dtype=float)
            self._u_prev = float(self.us[0])  

        xhat_pred = (self._Aaug @ self._xhat.reshape(-1, 1) +
                     self._Baug * self._u_prev +
                     self._caug).reshape(-1)

        y_hat = float(self._Caug @ xhat_pred.reshape(-1, 1))
        innovation = y_meas - y_hat
        self._xhat = xhat_pred + (self._L.reshape(-1) * innovation)

        x_hat = float(self._xhat[0])
        d_hat = float(self._xhat[1])

        self._log_dhat.append(d_hat)
        self._log_vzhat.append(x_hat)

        xs_new, us_new = self._compute_target(d_hat, x_ref)

        self._mpc_x0.value = np.array([x_hat])
        self._mpc_xs.value = np.array([xs_new])
        self._mpc_us.value = np.array([us_new])
        self._mpc_d.value = np.array([d_hat])

        self._mpc_prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
 
        if self._mpc_prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            u_applied = float(self._mpc_u.value[0, 0])
            x_traj = np.array(self._mpc_x.value, dtype=float)
            u_traj = np.array(self._mpc_u.value, dtype=float)
        else:
            u_applied = us_new
            x_traj = np.zeros((1, self.N + 1))
            u_traj = np.zeros((1, self.N))
            x_traj[0, 0] = x_hat

        u_applied = np.clip(u_applied, float(self.u_lb[0]), float(self.u_ub[0]))
        self._u_prev = u_applied

        return np.array([u_applied], dtype=float), x_traj, u_traj
