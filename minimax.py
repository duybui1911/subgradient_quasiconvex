import numpy as np
import cvxpy as cp
import scipy.optimize as optimize
import matplotlib.pyplot as plt

from problems.ex2_4_GMOP import Ex2_4

class Algorithm31():
    def __init__(self, fn, lower, upper, alpha, theta, eps, sigmax_conf=1, max_iterations=1000):
        self.alpha = alpha
        self.theta = theta
        self.max_iterations = max_iterations
        self.lower = lower
        self.upper = upper
        self.fn = fn
        self.list_xk = []
        self.list_yk = []
        self.fn_values = []
        self.sigma_conf = sigmax_conf
        self.eps = eps

    def solve(self, x0, min_loop=10):
        self.dim = len(x0)
        x_k = x0
        self.fn_values.append(max(Ex2_4.fn1(x0), Ex2_4.fn2(x0)))
        for k in range(self.max_iterations):
            # Step 1: Find y^k.
            y_k = self.find_y_k(
                x_k=x_k,
                p_k=self.get_p_k(k)
            )
            self.list_xk.append(x_k)
            self.list_yk.append(y_k)
            if np.linalg.norm(x_k - y_k) < self.eps and k >= min_loop:
                break
            z_km, m = self.find_smallest_positive_m(
                k, x_k, y_k
            )
            g_k = Ex2_4.get_grad(z_km, x_k)
            g_k /= np.linalg.norm(g_k)
            update_x_k = x_k - self.get_sigma_k(k)*g_k
            if not Ex2_4.check_constraints(update_x_k):
                update_x_k = self.find_projection(update_x_k)

            if np.linalg.norm(x_k - update_x_k) < self.eps and k >= min_loop:
                break
            x_k = update_x_k
            self.fn_values.append(max(Ex2_4.fn1(x_k), Ex2_4.fn2(x_k)))

        self.list_xk.append(x_k)
        return x_k, k

    def find_smallest_positive_m(self, k, x_k, y_k):
        MAX_M = 2000
        for m in range(1, MAX_M):
            # Check equation condition
            z_km = (1 - self.theta**m)*x_k + (self.theta**m)*y_k
            check_eq_4 = ((self.fn(z_km, x_k) - self.fn(z_km, y_k))
                          >= (self.alpha * np.linalg.norm(y_k - x_k)**2)/(2*self.get_p_k(k))
            )
            if check_eq_4:
                break

        return z_km, m

    def find_y_k(self, x_k, p_k):
        def obj(x):
            return self.fn(x_k, x) + (np.linalg.norm(x_k - x) ** 2) / (2*p_k)
        constraints, bounds = Ex2_4.get_constraints_scipy()
        res = optimize.minimize(
            obj,
            x0=np.ones(self.dim),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        return res.x

    def find_projection(self, x_k):
        # x_prj = cp.Variable(self.dim)
        # constraints = Ex2_4.get_constraints_cp(x_prj)
        # func = cp.norm2(x_k - x_prj)**2
        # objective = cp.Minimize(func)
        # prob = cp.Problem(
        #     objective=objective,
        #     constraints=constraints
        # )
        # result = prob.solve()
        def obj(x):
            return (np.linalg.norm(x_k - x) ** 2)
        constraints, bounds = Ex2_4.get_constraints_scipy()
        res = optimize.minimize(
            obj,
            x0=np.ones(self.dim),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        return res.x

    @staticmethod
    def get_p_k(k):
        return 1.

    def get_sigma_k(self, k):
        return self.sigma_conf/(k+1)

    def plot_results(self):
        self.list_xk = np.array(self.list_xk)
        self.list_yk = np.array(self.list_yk)

        delta_x = np.linalg.norm(
            self.list_xk[:-1] - self.list_xk[1:], axis=1
        )
        delta_xy = np.linalg.norm(
            self.list_xk[:-1] - self.list_yk, axis=1
        )
        plt.plot(delta_x, label=r'$||x^{k+1}-x^k||$', marker='o')
        plt.plot(delta_xy, label=r'$||x^k - y^k||$', marker='*')
        plt.title(r"Errors of $\alpha=$ {}, $\theta=$ {}, $\sigma=$ {}/(k+1)".format(
            self.alpha, self.theta, self.sigma_conf))
        plt.xlabel('Num of iterations')
        plt.ylabel('Errors')

        plt.legend()

        plt.savefig("Err_alpha={}_theta={}_sigma={}.png".format(
            self.alpha, self.theta, self.sigma_conf))
        plt.clf()

    def plot_tmp_solution(self):
        for i in range(self.dim):
            legend_txt = r'$x_{}$'.format(i)
            plt.plot(self.list_xk[:, i], label=legend_txt, marker='*')
        plt.title(r"$\alpha=$ {}, $\theta=$ {}, $\sigma=$ {}/(k+1)".format(
            self.alpha, self.theta, self.sigma_conf))
        plt.xlabel('Num of iterations')
        plt.ylabel('Value')

        plt.legend()

        plt.savefig("Sol_alpha={}_theta={}_sigma={}.png".format(
            self.alpha, self.theta, self.sigma_conf))
        plt.clf()

    def plot_tmp_values(self):
        self.fn_values = np.array(self.fn_values)
        legend_txt = r'$f(x^{k})$'
        plt.plot(self.fn_values, label=legend_txt, marker='*')
        plt.title(r"$\alpha=$ {}, $\theta=$ {}, $\sigma=$ {}/(k+1)".format(
            self.alpha, self.theta, self.sigma_conf))
        plt.xlabel('Num of iterations')
        plt.ylabel('Value')

        plt.legend()

        plt.savefig("Fx_alpha={}_theta={}_sigma={}.png".format(
            self.alpha, self.theta, self.sigma_conf))
        plt.clf()
if __name__ == "__main__":
    np.random.seed(1)
    fx = []
    numter = []
    for i in range(1, 10):
        alpha = 0.1 * i
        solver = Algorithm31(
            Ex2_4.func,
            lower=0,
            upper=2,
            alpha=alpha,
            theta=0.95,
            eps=1e-3,
            sigmax_conf=2.,
            max_iterations=200
        )
        x0 = np.array([np.random.uniform(0, 2), np.random.uniform(0, 2)])
        x_n, k = solver.solve(x0=x0, min_loop=0)
        numter.append([alpha, k])
        fx.append(
            [Ex2_4.fn1(x_n), Ex2_4.fn2(x_n)]
        )
        print(solver.fn_values[-1], solver.list_xk[-1])
        solver.plot_results()
        solver.plot_tmp_solution()
        solver.plot_tmp_values()

    fx = np.array(fx)
    print(fx[:,1])
    plt.plot(fx[:, 0], fx[:, 1], marker="*")
    plt.xlabel('Alpha')
    plt.ylabel('Num of iterations')
    plt.savefig("pareto.png")
