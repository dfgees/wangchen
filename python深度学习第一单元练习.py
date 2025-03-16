#第一单元
#1.1生成Numpy数组
# import cv2
# from matplotlib import pyplot as plt
#
# # 读取图像
# img = cv2.imread(r"C:\Users\111111\Desktop\20210721093747.jpg")
#
# # 检查图像是否成功加载
# if img is None:
#     print("Failed to load image. Check the file path.")
# else:
#     # 将图像从BGR转换为RGB
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     # 显示图像
#     plt.imshow(img_rgb)
#     # plt.axis('off')  # 关闭坐标轴
#     plt.show()
#
#     # 打印图像的数据类型和形状
#     print("数据类型：{}，形状：{}".format(type(img), img.shape))
#
# #数组属性
# print("img数组的维度：",img.ndim)
# print("img数组的形状",img.shape)
# print("img数组的数据类型:",img.dtype)

#将列表转换成ndarray
import numpy as np
# a=[3.14,2.17,0,1,2]
# nd1=np.array(a)
# print(nd1)
# print(type(nd1))
#
# #将嵌套列表转换为多维数组
# b=[[3.14,2.17,0,1,2],[1,2,3,4,5]]
# nd2=np.array(b)
# print(nd2)
# print(type(nd2))
# np.random.seed(2019)
# c=np.random.random([10])
# print(c)
# print(c[3])
# print(c[1:4])
# print(c[0:6:2])
# print(c[::-2])
# #截取一个多维数组的某个区域内的数据
# d=np.arange(25).reshape(5,5)
# print(d)
# print(d[3:,2:])
# print(d[2::2,::2])#   2::2 表示从第2行（索引为2）开始，每隔一行取一行（步长为2），::2 表示从第0列开始，每隔一列取一列（步长为2）。


from numpy import random as nr
# e=np.arange(1,25,dtype=float)
# print(e)
# e1=nr.choice(e,size=(3,4))#size指定输出数组形状,随机抽取可以重复
# e2=nr.choice(e,size=(3,4),replace=False)#replace默认为True,即可重复抽取，随机但不重复抽取
# print(e1)
# print(e2)
# e3=nr.choice(e,size=(3,4),p=e/np.sum(e))#下式参数p指定为每个元素对应的抽取概率，默认为每个元素被抽取的概率相同
# print(e3)
#
#
# A=np.array([[1,2,3],[4,5,6],[7,8,9]])
# B=np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(np.multiply(A,B))
# print(A*B)
#
# X=np.random.rand(2,3)
# def sigmoid(x):
#     return 1/(1+np.exp(-x))
# def relu(x):
#     return np.maximum(0,x)
# def softmax(x):
#     return np.exp(x)/np.sum(np.exp(x))
# print(f"输入参数X的形状：{X.shape}")
# print(f"激活函数sigmoid输出形状：{sigmoid(X).shape}")
# print(f"激活函数relu输出形状：{relu(X).shape}")
# print(f"激活函数softmax输出形状：{softmax(X).shape}")
#
# x1=np.array([[1,2,3],[4,5,6],[7,8,9]])
# x2=np.array([[1,2,3],[4,5,6],[7,8,9]])
# x3=np.dot(x1,x2)#                            点乘
# print(x3)
# x4=np.dot(x1,x2.T)#                           x2转置然后点乘
# print(x4)
#
# #  1）reshape函数
# arr=np.arange(10)
# print(arr)
# #将向量arr变换为2行5列
# print(arr.reshape(2,5))
# #指定维度时可以只指定行数或列数，其他用-1代替
# print(arr.reshape(5,-1))
# print(arr.reshape(-1,5))
#
# #  2)resize函数
# arr=np.arange(10)
# print(arr)
# #将向量变换为2行5列
# arr.resize(2, 5)
# print(arr)

# 3）T函数
arr=np.arange(12).reshape(3,4)
#将向量arr转置为3行4列
print(arr)
print(arr.T)

# 4)ravel函数
arr=np.arange(6).reshape(2,-1)
print(arr)
#按照列优先，展平
print(arr.ravel('F'))
#按照行优先，展平
print(arr.ravel())

# 5)flatten(order='C')函数
#把矩阵转化为向量，展平方式默认是行优先，这种需求经常出现在卷积网络与全连接层之间
a=np.floor(10*np.random.random((3,4)))
print(a)
print(a.flatten(order='C'))

# 6)squeeze函数
#squeeze函数主要用于降维，可以把矩阵中含1的维度去掉
arr=np.arange(3).reshape(3,1)
print(arr.shape)    #（3，1）
print(arr.squeeze().shape)#(3,)
arr1=np.arange(6).reshape(3,1,2,1)
print(arr1.shape)#(3,1,2,1)
print(arr1.squeeze().shape)#(3,2)

# 7)transpose函数
#transpose函数主要用于对高维矩阵进行轴对换，经常用于深度学习中，比如把图像表示颜色的RGB顺序改为GBR的顺序
arr2=np.arange(24).reshape(2,3,4)
print(arr2.shape)
print(arr2.transpose(1,2,0).shape)

#合并数组
"""
np.append             内存占用大
np.concatenate        没有内存问题
np.stack              沿着新的轴加入一系列数组
np.hstack             栈数组垂直顺序
np.vstack             栈数组垂直顺序
np.datack             栈数组按顺序深入
np.vsplit             将数组分解成垂直的多个子数组的列表
zip([iterable,...])   将对象中对应的元素打包成一个个元组构成的zip对象  

"""
#  1.append
a=np.array([1,2,3])
b=np.array([4,5,6])
#合并一维数组
c=np.append(a,b)
print(c)
#合并二维数组
a=np.arange(4).reshape(2,2)
b=np.arange(4).reshape(2,2)
print(a)
print(b)
#按行合并
c=np.append(a,b,axis=0)
print(c)
print(c.shape)
#按列合并
d=np.append(a,b,axis=1)
print(d)
print(d.shape)

#  2.concatenate
a=np.array([[1,2],[3,4]])
b=np.array([[5,6]])
c=np.concatenate((a,b),axis=0)
print(c)
d=np.concatenate((a,b.T),axis=1)
print(d)

#  3.Stack
a=np.array([[1,2],[3,4]])
b=np.array([[5,6],[7,8]])
print(np.stack((a,b),axis=0))

#  4.zip
#zip是Python的一个内置函数，多用于张量计算中
a=np.array([[1,2],[3,4]])
b=np.array([[5,6],[7,8]])
c=c=zip(a,b)
for i,j in c:
    print(i,end=",")
    print(j)
#使用zip函数组合两个向量
a1=[1,2,3]
a2=[4,5,6]
c1=zip(a1,a2)
for i,j in c1:
    print(i,end=",")
    print(j)

#      批处理
"""

1.得到数据集
2.随机打乱数据
3.定义批大小
4.批处理数据集

"""
#举例
#生成10000个形状为2*3的矩阵
data_train=np.random.randn(10000,2,3)
#这是一个3维矩阵，第一个维度为样本数，后两个是数据形状
print(data_train.shape)
#打乱这10000条数据
np.random.shuffle(data_train)
#定义批量大小
batch_size=100
#进行批处理
for i in range(0,len(data_train),batch_size):
    x_train_sum=np.sum(data_train[i:i+batch_size])
    print("第{}批次，该批次的数据之和：{}".format(i,x_train_sum))
#节省内存
"""
cumsum、cumproduct累计求和，求积
mean计算均值
median计算中位数
std计算标准差
corrcoef计算相关系数

math与numpy函数比numpy函数更快

"""

#循环与向量运算比较
import time
x1=np.random.rand(1000000)
x2=np.random.rand(1000000)
##使用循环计算向量点积
tic=time.process_time()
dot=0
for i in range(1000000):
    dot+=x1[i]*x2[i]
toc=time.process_time()
print("dot="+str(dot)+"\n for loop-----Computation time ="+str(1000*(toc-tic))+"ms")
x1=np.random.rand(1000000)
x2=np.random.rand(1000000)
##使用numpy函数求点积
tic=time.process_time()
dot=0
dot=np.dot(x1,x2)
toc=time.process_time()
print("dot="+str(dot)+"\n for verctor version----Computation time ="+str(1000*(toc-tic))+"ms")

#广播机制
A=np.arange(0,40,10).reshape(4,1)
B=np.arange(0,3)
print(A)
print(B)
print("A矩阵的形状：{}，B矩阵的形状：{}".format(A.shape,B.shape))
c=A+B
print(c)
print("C矩阵的形状：{}".format(c.shape))


