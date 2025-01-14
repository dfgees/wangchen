"""
拉格朗日插值多项式的应用习题
"""

import math
xn=[(math.pi/6, 1/2),(math.pi/4, 1/math.sqrt(2)),(math.pi/3, math.sqrt(3)/2)]

x=None
def Lagrange1(flag,x):
    """
    拉格朗日一次多项式
    """
    if flag==1:
        Lagrange=(x-xn[1][0])/(xn[0][0]-xn[1][0])*xn[0][1]+(x-xn[0][0])/(xn[1][0]-xn[0][0])*xn[1][1]
    elif flag==2:
        Lagrange=(x-xn[2][0])/(xn[1][0]-xn[2][0])*xn[1][1]+(x-xn[1][0])/(xn[2][0]-xn[1][0])*xn[2][1]
    return Lagrange

def Lagrange2(x):
    """
    拉格朗日二次多项式
    """
    Lagrange=((x-xn[1][0])*(x-xn[2][0]))/((xn[0][0]-xn[1][0])*(xn[0][0]-xn[2][0]))*xn[0][1]+((x-xn[0][0])*(x-xn[2][0]))/((xn[1][0]-xn[0][0])*(xn[1][0]-xn[2][0]))*xn[1][1]+((x-xn[0][0])*(x-xn[1][0]))/((xn[2][0]-xn[0][0])*(xn[2][0]-xn[1][0]))*xn[2][1]
    return Lagrange

x=float(input("请输入sin（x）中x的值:"))*(math.pi/180)


Lagrange1(1,x)
print(f'利用x0，x1计算出来的拉格朗日一次多项式的值为{Lagrange1(1,x)}')
Lagrange1(2,x)
print(f'利用x1，x2计算出来的拉格朗日一次多项式的值为{Lagrange1(2,x)}')
Lagrange2(x)
print(f'利用x0,x1,x2计算出来的拉格朗日二次多项式的值为{Lagrange2(x)}')

def R1(flag,x):
    if flag==1:
        R1_left=(x-xn[0][0])*(x-xn[1][0])*(-1/2)/2
        R1_right=(x-xn[0][0])*(x-xn[1][0])*(-1/math.sqrt(2))/2
        error1=math.sin(x)-Lagrange1(1,x)
        if error1>R1_left and error1<R1_right:
            print(f"计算的值在误差范围内,实际误差为{error1}")
        else:
            print(f"计算的值超出了误差范围内，精确度不够，实际误差为{error1}")

    if flag==2:
        R1_left=(x-xn[1][0])*(x-xn[2][0])*(-1/math.sqrt(2))/2
        R1_right=(x-xn[1][0])*(x-xn[2][0])*(-math.sqrt(3)/2)/2
        error2 = math.sin(x) - Lagrange1(2, x)
        if error2>R1_left and error2<R1_right:
            print(f"计算的值在误差范围内,实际误差为{error2}")
        else:
            print(f"计算的值超出了误差范围内，精确度不够，实际误差为{error2}")

def R2(x):
    R2_left=(x-xn[0][0])*(x-xn[1][0])*(x-xn[2][0])*(-1/2)/6
    R2_right=(x-xn[0][0])*(x-xn[1][0])*(x-xn[2][0])*(-math.sqrt(3)/2)/6
    error3 = math.sin(x)-Lagrange2(x)
    if error3>R2_left and error3<R2_right:
        print(f"计算的值在误差范围内,实际误差为{error3}")
    else:
        print(f"计算的值超出了误差范围内，精确度不够，实际误差为{error3}")
R1(1,x)
R1(2,x)
R2(x)

