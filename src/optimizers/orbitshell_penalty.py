import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
def softplus(z, k=50.0):
    # numerically stable softplus
    return ca.log1p(ca.exp(k*z)) / k
    #return ca.if_else(k * z > 50, z, ca.log1p(ca.exp(k * z)) / k)

def shell_penalty(r, a, b, s=1, k=5):
    """"
    Smooth function with slope -> flat -> slope.
    If opposite=True, the second slope is negative.
    """
    x = 2*(r-a)/(b-a)-1
    x_a = -1
    x_b = 1

    f_raw = -(s*x - s*softplus(x-x_a, k) - s*softplus(x-x_b, k))

    # Compute flat value: function at the center of the flat region
    flat_val = -(s*(x_a+x_b)/2 - s*softplus((x_a+x_b)/2 - x_a, k)
                 - s*softplus((x_a+x_b)/2 - x_b, k))

    return f_raw - flat_val
r = np.arange(22,31)
y = [shell_penalty(i,22,30) for i in r]
plt.plot(r,y)
plt.grid()
plt.savefig("figurareward")
plt.show()