"""

              对于函数z=x2+y2，请用梯度下降法求出使函数取得最小值的x、y 值。


"""
x = 3.0   #初始x的值
y = 2.0   #初始y的值
alpha =0.1   #（学习率）控制参数更新的步长，值过大会导致震荡，过小则收敛缓慢
max_number = 1000   #最大循环次数
tolerance =1e-6   #设定最小的值

for i in range(max_number):
    dx = 2 * x
    dy = 2 * y
    if abs(dx) < tolerance and abs(dy) < tolerance:
        break

    x = x -alpha * dx
    y = y -alpha * dy
print(f"迭代次数为{i+1}")
print(f"最小值点{x:.6f},{y:.6f}")