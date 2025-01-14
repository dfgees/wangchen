"""
具有四个节点的插值型求积公式，至少有3次代数精度
"""
from sympy import *
#求Ak(k=0,1,2,3)
x=Symbol('x')
A0= integrate((x-1)*(x-2)*(x-3)/(-6),(x, 0, 3))
print(A0)
A1= integrate((x-0)*(x-2)*(x-3)/2,(x, 0, 3))
print(A1)
A2= integrate((x-0)*(x-1)*(x-3)/(-2),(x, 0, 3))
print(A2)
A3= integrate((x-0)*(x-1)*(x-2)/6,(x, 0, 3))
print(A3)
print(f"f(x)在0~3上的积分近似于{A0}f(x0)+{A1}f(x1)+{A2}f(x2)+{A3}f(x3)")


