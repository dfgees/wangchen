a=[(1,1,1,6),(0,4,-1,5),(2,-2,1,1),]
a[2]=tuple(a1*(-2) + a3 for a1, a3 in zip(a[0], a[2]))#将a[0]+a[2]的值赋给a[2]
print(a)
a[2]=tuple(a2 + a3 for a2, a3 in zip(a[1], a[2]))#将a[1]+a[2]的值赋给a[2]
print(a)
a[2]=tuple([num/(-2)for num in a[2]])#给a[2]/(-2)得到    x3    的值
print(a)
a[1]=tuple(a2 + a3 for a2, a3 in zip(a[1], a[2]))#计算a[1]+a[2]得到4*x2的值
print(a)
a[1]=tuple([num/4 for num in a[1]])#给a[1]/4得到    x2    的值
print(a)
a[0]=tuple((-1)*a2 + (-1)*a3 + a1 for a2, a3, a1 in zip(a[1], a[2], a[0]))#计算a[0]-a[1]-a[2]得到    x1    的值
print(a)
print(f"利用消元法解得  x1={a[0][3]},  x2={a[1][3]},  x3={a[2][3]}")