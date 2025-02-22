def f(x):
    return x**3-x-1

a=1.0
b=1.5

if f(a)*f(b)>0:
    print(f"所选的区间错误，在此区间内没有零点")
    exit()

count=0
fa=f(a)
fb=f(b)

while(b-a)>0.001:
    count+=1
    c=(a+b)/2
    fc=f(c)

    if fc==0:
        break
    if(fa*fc<0):
        b=c
    else:
        a=c
        fa=fc

number=(a+b)/2
print(f'精确值为{number}')