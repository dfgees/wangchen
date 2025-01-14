from sympy import *

x=Symbol('x')
y=exp(x)
#先计算（Pk，f）（k=0，1，2，3)

#计算内积
neiji=[None,None,None,None]
neiji[0] = integrate(y, (x, -1, 1)).evalf()
neiji[1] = integrate(x * y, (x, -1, 1)).evalf()
neiji[2] = integrate((3*(x**2)-1)/2*y,(x,-1,1)).evalf()
neiji[3] = integrate((5*(x**3)-3*x)/2*y, (x, -1, 1)).evalf()
# neiji[0]=round(neiji[0],4)
# neiji[1]=round(neiji[1],4)
# neiji[2]=round(neiji[2],5)
# neiji[3]=round(neiji[3],5)

for i in range(4):
    print(f"neiji[{i}]={neiji[i]}")
#计算系数
a=[None,None,None,None]
a[0]=neiji[0]/2
a[1]=neiji[1]/2*3
a[2]=neiji[2]/2*5
a[3]=neiji[3]/2*7
# a[0]=round(a[0],4)
# a[1]=round(a[1],5)
# a[2]=round(a[2],4)
# a[3]=round(a[3],5)
print(f'a0={a[0]}\na1={a[1]}\na2={a[2]}\na3={a[3]}\n')

s=a[0]*1+a[1]*x+a[2]*((3*(x**2)-1)/2)+a[3]*((5*(x**3)-3*x)/2)
print(f"s={s}")
#计算平方误差
s1=0
for i in range(4):
    s1+=(2/(2*i+1))*((a[i])**2)
s1=round(s1,6)
print(f"s1={s1}")
error=integrate(y**2, (x, -1, 1))
error=round(error,6)
print(f"error={error}")
error-=s1
error=round(error,6)
print(f"error={error}")






