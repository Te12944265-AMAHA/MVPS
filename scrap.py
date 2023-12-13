import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize as jminimize


def fun_rosenbrock(x):
    return 10.0 * (x[1] - x[0]**2)

x0 = np.array([2.0, 2.0])
x0 = jnp.asarray(x0)

opts = {'maxiter': 10}
res = jminimize(fun_rosenbrock, x0, method="BFGS", options=opts)
print(res.success)
print(res.status)
print(res.x)
print(res.fun)
print(res.nit)