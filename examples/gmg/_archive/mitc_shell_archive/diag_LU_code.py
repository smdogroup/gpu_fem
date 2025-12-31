# see whether diag LU factor can be done accurately with cusparse vals instead..
# writing my own LU factor here..
# maybe could do cholesky instead here?

import numpy as np
import matplotlib.pyplot as plt

# to get D^-1 * vec
# compare computing linear operator for Dinv * v 6x6 with solving U^-1 L^-1 * v for numerical stability..
LU_v1 = np.array([2.862581507006e+10, -2.963357073384e+09, -1.526220420419e+08, 1.405078689269e+02, -6.295366772190e+02, -1.623092911077e+04, 
-1.035204435623e-01, 2.712770127673e+10, -6.162658569818e+07, 2.562720308072e+04, -2.056778443632e+02, -1.602016522188e+04, 
-5.331622581518e-03, -2.271721627628e-03, 4.344895589633e+10, 5.886114374825e+01, -1.930481620788e+02, -1.229304815314e+02, 
4.908432077136e-09, 9.446876025102e-07, 1.354719406577e-09, 1.735059344926e+07, -2.310436248291e+06, -2.287906994574e+05, 
-2.199192147667e-08, -7.581838293846e-09, -4.443102442601e-09, -1.331617996265e-01, 1.615988392843e+07, -9.801672306930e+04, 
-5.670032126962e-07, -5.905463591795e-07, -2.829308069558e-09, -1.318633279758e-02, -6.065434844915e-03, 2.094792210936e+07]).reshape((6,6))

L = LU_v1.copy()
U = LU_v1.copy()
for i in range(6):
    for j in range(6):
        if i > j: U[i,j] = 0.0
        if i < j: L[i,j] = 0.0
        if i == j: L[i,j] = 1.0

# implement triangular solves myself.. on 6x1 vecs
b = np.random.rand(6)
x = np.zeros_like(b)
y = np.zeros_like(b)

# L^-1 * b => y, forward triang solve
btemp = b.copy()
for i in range(6):
    y[i] = btemp[i] / L[i,i]
    for j in range(i+1,6):
        btemp[j] -= L[j,i] * y[i]

# check resid on Ly - b
Ry = L @ y - b
print(f"{np.linalg.norm(Ry)=:.2e}")

# U^-1 * y => x, backwards triang solves
ytemp = y.copy()
for i in range(5,-1,-1):
    x[i] = ytemp[i] / U[i,i]
    for j in range(i):
        ytemp[j] -= U[j,i] * x[i]

Rx = U @ x - y
print(f"{np.linalg.norm(Rx)=:.2e}")

# now compare with computing an effective Dinv matrix..
Dinv = np.zeros_like(LU_v1)

# Dinv * e1 => out used to construct Dinv matrix..
for v in range(6):
    ev = np.zeros(6)
    ev[v] = 1.0

    temp = ev.copy()
    y = np.zeros_like(temp)
    for i in range(6):
        y[i] = temp[i] / L[i,i]
        for j in range(i+1,6):
            temp[j] -= L[j,i] * y[i]

    # U^-1 * y => x, backwards triang solves
    ytemp = y.copy()
    xtemp = np.zeros_like(temp)
    for i in range(5,-1,-1):
        xtemp[i] = ytemp[i] / U[i,i]
        for j in range(i):
            ytemp[j] -= U[j,i] * xtemp[i]
    
    Dinv[:,v] = xtemp

plt.imshow(Dinv)
plt.show()

# now compare Dinv @ b => x2 with x from triang solves..
x2 = np.dot(Dinv, b)
x_diff = x2 - x
x_diff_nrm = np.linalg.norm(x_diff)
x_nrm = np.linalg.norm(x)

print(f"{x_diff_nrm=:.2e} {x_nrm=:.2e}")
