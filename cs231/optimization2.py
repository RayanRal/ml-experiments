import math

x = -2; y = 5; z = -4

q = x + y
f = q * z

# backpropagation
dfdz = q
dfdq = z

dqdx = 1.0
dqdy = 1.0

dfdx = dfdq * dqdx
dfdy = dfdq * dqdy



x = 3; y = -4

#forward pass
sigy = 1.0 / (1 + math.exp(-y))  #1
num = x + sigy                   #2
sigx = 1.0 / (1 + math.exp(-x))  #3
xpy = x + y
xpysqr = xpy ** 2
den = sigx + xpysqr
invden = 1 / den
f = num * invden


#backward pass
dnum = invden
dinvden = num
dden = (-1.0 / (den ** 2)) * dinvden
dsigx = 1 * dden
dxpysqr = 1 * dden
dxpy = (2 * xpy) * dxpysqr
dx = 1 * dxpy
dy = 1 * dxpy
dx += ((1 - sigx) * sigx) * dsigx # very important addition! dx +=
dx += (1) * dnum
dsigy = 1 * dnum
dy += ((1 - sigy) * sigy) * dsigy