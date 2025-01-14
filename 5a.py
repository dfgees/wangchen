X=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
y=[1,None,None,None,None,None,None,None,None,None,None]
h=0.1
#欧拉法
for i in range(0,10):
    index=i+1
    y[index]=y[i]+h*(y[i]-(2*X[i])/y[i])
    y[index]=round(y[index],4)
    print(f"y[{index}]={y[index]}")
#隐式欧拉法
for i in range(0,10):
    index=i+1
    a=y[i]+h*(y[i]-(2*X[i])/y[i])
    y[index]=y[i]+h*(a-(2*X[index])/a)
    y[index]=round(y[index],4)
    print(f"y[{index}]={y[index]}")

#改进欧拉法
for i in range(0,10):
    index=i+1
    a=y[i]+h*(y[i]-(2*X[i])/y[i])
    b=y[i]+h*(a-(2*X[index])/a)

    y[index]=(a+b)/2
    y[index]=round(y[index],4)
    print(f"y[{index}]={y[index]}")

#龙格——库塔法
for i in (0,2,4,6,8):
    h=0.2
    k1=y[i]-2*X[i]/y[i]
    k2=y[i]+h/2*k1-((2*X[i]+h)/(y[i]+h/2*k1))
    k3=y[i]+h/2*k2-((2*X[i]+h)/(y[i]+h/2*k2))
    k4=y[i]+h*k3-((2*(X[i]+h))/(y[i]+h*k3))
    index=i+2
    y[index]=y[i]+h*(k1+2*k2+2*k3+k4)/6
    y[index]=round(y[index],5)
    print(f"y[{index}]={y[index]}")