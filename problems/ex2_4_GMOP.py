import numpy as np
from autograd import grad
import cvxpy as cp

class Ex2_4():
    def func(x, y):
        return np.max(
           Ex2_4.vector_func(x, y)
        )

    def vector_func(x, y):
        vector =  [
            Ex2_4.fn1(y) - Ex2_4.fn1(x),
            Ex2_4.fn2(y) - Ex2_4.fn2(x)
        ] 
        return vector
 
    def get_grad(z_k, x_k):
        i = np.argmax(Ex2_4.vector_func(z_k, x_k))
        func_grad = Ex2_4.d_fn1 if i == 0 else Ex2_4.d_fn2

        return func_grad(x_k)

    def get_fn_i(i):
        if i == 1:
            return Ex2_4.fn1
        if i == 2:
            return Ex2_4.fn2

    def d_fn1(x):
        return np.array([(x[0]-1), 2*(x[1] - 5)])

    def d_fn2(x):
        return np.array([2*(x[0] - x[1]), -2*(x[0] - x[1])])

    def fn1(x):
        return 0.5*(x[0]-1)**2 + (x[1] - 5)**2

    def fn2(x):
        return (x[0] - x[1])**2
    
    def g1(x):
        return x[0]**2 + x[1]**2 - 4

    def g2(x):
        return -x[0]

    def g3(x):
        return -x[1]

    def get_constraints_scipy():
        constraints = [
            {"type": "ineq", "fun": lambda x: -Ex2_4.g1(x)},  # g1(x) <= 0
            {"type": "ineq", "fun": lambda x: -Ex2_4.g2(x)},  # g2(x) <= 0
            {"type": "ineq", "fun": lambda x: -Ex2_4.g3(x)},  # g3(x) <= 0
        ]
        bounds = [(0., 2.), (0., 2.)]
        return constraints, bounds
    
    def check_constraints(x):
        check_rs = Ex2_4.g1(x) <= 0 and Ex2_4.g2(x) <= 0 and Ex2_4.g3(x) <= 0
        return check_rs