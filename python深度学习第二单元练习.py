"""
1. requires_grad 是什么？
   requires_grad 是 PyTorch 张量（torch.Tensor）的一个布尔属性，用于控制是否对该张量进行梯度跟踪。简单来说：

   requires_grad=True：PyTorch 会记录该张量的所有操作，以便后续自动计算梯度（用于反向传播）。

   requires_grad=False（默认值）：不跟踪该张量的操作，无法计算其梯度。

2. 为什么需要设置 requires_grad？
   在训练神经网络时，我们需要通过反向传播计算模型参数的梯度（如权重 w 和偏置 b），并使用梯度下降等优化算法更新这些参数。requires_grad 的作用是：

   标记需要优化的参数：例如，将权重和偏置设置为 requires_grad=True。

   构建动态计算图：PyTorch 会跟踪所有涉及 requires_grad=True 张量的操作，形成计算图以支持自动微分。

3.  torch.linspace() 是 PyTorch 中的一个函数，用于生成一个一维张量（Tensor），
包含在指定区间内均匀间隔的数值。它的核心功能是生成等间距的数值序列，类似于数学中的线性空间（Linear Space）。

4.  在 PyTorch 中，dtype=dtype 的作用是 显式指定张量的数据类型。这里的 dtype 是一个变量（需提前定义），它表示你想让张量 w 存储的数据类型。
以下分步解释：
      1. dtype 的作用
      dtype 是张量（Tensor）的数据类型，决定了张量中元素的存储方式和计算精度。

      常见的数据类型：

                    torch.float32（默认）：32 位浮点数（单精度）。

                    torch.float64：64 位浮点数（双精度）。

                    torch.int32：32 位整数。

                    torch.bool：布尔类型（True/False）。
5.  .mm() 是 PyTorch 中用于矩阵乘法的方法，即矩阵与矩阵相乘。

6.  x=torch.unsqueeze(）
      在 PyTorch 里，torch.unsqueeze() 函数能够在指定维度上给张量插入一个新的维度。
      下面详细介绍它的用法、参数以及结合具体示例说明如何使用 torch.unsqueeze() 。
    函数原型：
            torch.unsqueeze(input, dim)
    代码解释
                 创建一维张量：x = torch.tensor([1, 2, 3, 4]) 创建了一个形状为 (4,) 的一维张量。
                 在第 0 维插入新维度：torch.unsqueeze(x, dim=0) 在第 0 维插入一个新维度，使得张量形状变为 (1, 4)。
                在第 1 维插入新维度：torch.unsqueeze(x, dim=1) 在第 1 维插入一个新维度，张量形状变为 (4, 1)。
                使用负索引插入新维度：torch.unsqueeze(x, dim=-1) 等同于 torch.unsqueeze(x, dim=1)，
                因为负索引从最后一个维度开始计数，这里在最后一个维度插入新维度，张量形状变为 (4, 1)。

参数说明
input：这是需要处理的输入张量。
dim：表示要插入新维度的位置。dim 的取值范围是 -input.dim() - 1 到 input.dim() ，负索引意味着从最后一个维度开始计数。
"""
import torch
from torch import nn

x=torch.tensor([1,2])
y=torch.tensor([3,4])
#不修改自身数据，x的数据不变，返回一个新的Tensor
z=x.add(y)
print(z)
print(x)
#修改自身数据，运算结果存在x中，x被修改
x.add_(y)
print(x)
"""

        创建Tensor

"""
#根据列表数据生成Tensor
t1=torch.Tensor([1,2,3,4,5,6])
print(t1)
#根据指定形状生成Tensor
t2=torch.Tensor(2,3)
print(t2)
#根据给定的Tensor的形状
t3=t=torch.Tensor([[1,2,3],[4,5,6]])
print(t3)
#查看Tensor的形状
print(t3.size())
#shape与size等价
t3.shape
#根据已有形状创建Tensor
torch.Tensor(t.size())


import torch
t1=torch.Tensor(1)
t2=torch.tensor(1)
print("t1的值{}，t1的数据类型{}".format(t1,t1.type()))
print("t2的值{}，t2的数据类型{}".format(t2,t2.type()))

#生成一个单位矩阵
t1=torch.eye(2,2)
#自动生成元素全是0的矩阵
t2=torch.zeros(2,3)
#根据规则生成数据
t3=torch.linspace(1,10,4)
#生成满足均匀分布随机数
t4=torch.rand(2,3)
#生成满足标准分布随机数
t5=torch.randn(2,3)
#返回多给数据形状相同，值全为0的张量
t6=torch.zeros_like(torch.randn(2,3))
print(t1)
print(t2)
print(t3)
print(t4)
print(t5)
print(t6)
# 生成一个2x3的矩阵
x=torch.randn(2,3)
#查看矩阵的形状
print(x.size())
#查看矩阵的维度
print(x.dim())
#把x变成一个3x2的矩阵
b=x.view(3,2)
print(b.size())
# 把x展平为一维向量
y=x.view(-1)
print(y.size())
z=torch.unsqueeze(y,0)
#查看z的形状
print(z.size())
#计算z的元素个数
print(z.numel())
###################################################################################################################
"""
索引操作
"""
#设置一个随机种子
torch.manual_seed(100)        #设置随机种子，确保每次运行代码时生成的随机数相同，便于调试和复现结果。
#生成一个形状为2x3的矩阵
x=torch.randn(2,3)            #生成的满足正态分布的随机数
"""
例如：x = [[-0.1234,  0.5678, -0.9101],
     [ 1.2345, -0.6789,  0.4321]]
"""
#根据索引获取第一行的所有数据
x[0,:]
#获取最后一列的数据
x[:,-1]
#生成是否大于0的Byter张量
mask=x>0
"""
作用：生成一个布尔张量，表示 x 中每个元素是否大于 0。

解释：mask 是一个与 x 形状相同的布尔张量，例如：

mask = [[False,  True, False],
        [ True, False,  True]]
"""
#获取大于0的值
torch.masked_select(mask,x)
#获取非0下标，即行、列索引
torch.nonzero(mask)
##获取指定索引对应的值，输出根据以下规则得到

#out[i][j]=input[index[i][j]][j]   #if  dim==0
#out[i][j]=input[i][index[i][j]]   #if  dim==1
index=torch.LongTensor([[0,1,1]])
torch.gather(x,0,index)
"""

index 是一个形状为 1x3 的张量，表示在每一列中选择的行索引。

torch.gather(x, 0, index) 的输出形状与 index 相同，规则为：

out[i][j] = input[index[i][j]][j]
例如，如果 x 为：

x = [[-0.1234,  0.5678, -0.9101],
     [ 1.2345, -0.6789,  0.4321]]
则 torch.gather(x, 0, index) 的结果为：

[[-0.1234, -0.6789,  0.4321]]
"""
index=torch.LongTensor([[0,1,1],[1,1,1]])
a=torch.gather(x,1,index)

"""

index 是一个形状为 2x3 的张量，表示在每一行中选择的列索引。

torch.gather(x, 1, index) 的输出形状与 index 相同，规则为：

out[i][j] = input[i][index[i][j]]
例如，如果 x 为：

x = [[-0.1234,  0.5678, -0.9101],
     [ 1.2345, -0.6789,  0.4321]]
则 torch.gather(x, 1, index) 的结果为：

[[-0.1234,  0.5678,  0.5678],
 [-0.6789, -0.6789, -0.6789]]
 
 """
#把a的值返回到一个2x3的0矩阵中
z=torch.zeros(2,3)
z.scatter_add(0,index,a)
"""

作用：将 a 的值根据索引 index 累加到全 0 矩阵 z 中。

解释：

z 是一个形状为 2x3 的全 0 矩阵。

scatter_add 的规则为：

z[index[i][j]][j] += a[i][j]  # 如果 dim == 0
例如，如果 index 和 a 为：
index = [[0, 1, 1], [1, 1, 1]]
a = [[-0.1234,  0.5678,  0.5678],
     [-0.6789, -0.6789, -0.6789]]
则 z.scatter_add(0, index, a) 的结果为：

z = [[-0.1234,  0.0000,  0.5678],
     [ 0.0000, -0.1111, -0.1111]]

"""

###################################################################################################################
"""
广播机制
"""
import numpy as np
A=np.arange(0,40,10).reshape(4,1)
print(A)
B=np.arange(0,3)
print(B)
#把ndarray转换为Tensor
A1=torch.from_numpy(A)         #形状为4x1
B1=torch.from_numpy(B)         #形状为3
#Tensor自动实现广播
c=A1+B1
print(c)
"""
广播规则：

维度对齐：将B1的形状从(3,)扩展为(1, 3)。

维度扩展：将A1的形状从(4, 1)扩展为(4, 3)，B1从(1, 3)扩展为(4, 3)。
"""
#我们可以根据广播机制手工进行配置
#根据规则1，B1需要向A1看齐，把B1变为（1，3）
B2=B1.unsqueeze(0)             #B2的形状为1x3
"""
将B1从(3,)变为(1, 3)
"""
print(B2)
#使用expend函数重复数组，分别转变为4x3的矩阵
A2=A1.expand(4,3)
B3=B2.expand(4,3)
"""
# 扩展A1和B2为4x3
A2 = A1.expand(4, 3)  # 从(4,1)扩展为(4,3)
B3 = B2.expand(4, 3)  # 从(1,3)扩展为(4,3)
"""
#然后进行相加，c1与c结果一致
#####################################################################################################################
"""
逐元素操作
"""
t=torch.randn(1,3)
t1=torch.randn(3,1)
t2=torch.randn(1,3)
#t+0.1*(t1/t2)
torch.addcdiv(t,t1,t2,value=0.1)
"""
t+0.1*(t1/t2)
"""
#计算sigmoid
torch.sigmoid(t)
#将t限制在[0,1]之间
torch.clamp(t,0,1)
#进行t+2运算
t.add_(2)
################################################################################################################
"""
归并操作
"""
"""
cumprod(t,axis)                      在指定维度对t进行累加
cumsum                               在指定维度对t进行累加
dist（a,b,p=2)                       返回a,b之间的p阶范数
mean/median                          均值/中位数
std/var                              标准差/方差
norm(t,p=2)                          返回t的p阶范数
prod（t）/sum(t)                      返回t所有元素的积/和

"""
a=torch.linspace(0,10,6)
"""
作用：生成一个包含 6 个元素的一维张量，数值范围从 0 到 10（均匀分布）。

输出：

a = tensor([ 0.0,  2.0,  4.0,  6.0,  8.0, 10.0]
"""
#使用view方法，把a变成2x3矩阵
a=a.view((2,3))
#沿y轴方向累加，即dim=0
b=a.sum(dim=0)
"""
作用：在维度 0（行方向）对每列进行求和。

计算过程：

第 0 列：0.0 + 6.0 = 6.0
第 1 列：2.0 + 8.0 = 10.0
第 2 列：4.0 + 10.0 = 14.0

结果形状：

b = tensor([6.0, 10.0, 14.0])  # 形状为 [3]

"""
#沿y轴方向累加，即dim=0,并保留含1的维度
b=a.sum(dim=0,keepdim=True)
"""
b的形状为[1,3]

作用：在维度 0 求和后，保留结果的维度信息。

结果形状：
b = tensor([[6.0, 10.0, 14.0]])  # 形状为 [1, 3]
"""
######################################################################################################
"""
比较操作
"""
"""
eq                                                   比较张量是否相等，支持广播机制
equal                                                比较张量是否有相同的形状与值
ge/le/gt/lt                                          大于/小于比较，大于或等于/小于或等于比较
max/min(t,axis)                                      返回最值，若指定axis，则额外返回下标
topk(t,k,axis)                                       在指定的axis维上取最高的k个值
"""
x=torch.linspace(0,10,6).view(2,6)
#求所有元素的最大值
torch.max(x)                                         #结果为10
#求y轴方向的最大值
torch.max(x,dim=0)                                   #结果为[6,8,10]
#求最大的元素
torch.topk(x,1,dim=0)                             #结果为[6,8,10]
###################################################################################################
"""
矩阵操作
"""
"""
dot(t1,t2)                                                 计算张量（1维）的内积（或点积）
mm(mat1,mat2)/bmm(batch1,batch2)                           计算矩阵乘法/含批量的3维矩阵乘法
mv（t1，t2）                                                计算矩阵与向量乘法
t                                                          转置
svd（t）                                                    计算t的SVD分解
"""
a=torch.tensor([2,3])
b=torch.tensor([3,4])
z=torch.dot(a,b)
print(z)
"""
作用：计算两个一维张量的点积（对应元素相乘后求和）。

公式：2*3 + 3*4 = 6 + 12 = 18。

条件：两个张量必须是一维且长度相同。
"""
x=torch.randint(10,(2,3))# 生成 2x3 的随机矩阵，元素值在 [0,10) 之间
y=torch.randint(6,(3,4))# 生成 3x4 的随机矩阵，元素值在 [0,6) 之间
z=torch.mm(x,y)
print(z)
"""
规则：

第一个矩阵的列数（3）必须等于第二个矩阵的行数（3）。

结果矩阵的形状为 (2, 4)。

示例：

x = [[1, 2, 3],
     [4, 5, 6]]
y = [[0, 1, 2, 3],
     [4, 5, 6, 7],
     [8, 9, 10, 11]]
z = [[1*0 + 2*4 + 3*8, ..., ...],
     [4*0 + 5*4 + 6*8, ..., ...]]  # 结果形状为 2x4
"""
x=torch.randint(10,(2,2,3))                        # 形状 (2,2,3)，表示批量大小为2，每个小矩阵为2x3
y=torch.randint(6,(2,3,4))                         # 形状 (2,3,4)，批量大小为2，每个小矩阵为3x4
z=torch.bmm(x,y)
print(z)                                           # 输出形状为 (2,2,4) 的张量
"""
torch.autograd包用来自动求导，torch.Tensor和torch.Function为autograd包的两个核心类，他们相互连接并生成一个有向非循环图。
"""
"""
PyTorch使用torch.autograd.backward来实现反向传播，backware函数的具体格式如下：
torch.autograd.backward(
    tensors,                                     #用于计算梯度的张量。
    grad_tensors=None,                           #用于计算非标量的梯度。其形状一般需要与前面的张量保持一致。
    retain_graph=None,                           #通常在调用一次backward函数后，PyTorch会自动销毁计算图，如果要想对某个变量重复调用backward函数。则需要将该参数设置为True
    create_graph=False,                          #当设置为True的时候可以用来计算更高阶的梯度
    grad_variables=None,                         #这个参数后面的版本中应该会丢弃，直接使用grad_tensors就好了
)

"""
"""
假设x、w、b都是标量，z=wx+b，对标量z调用backward函数。无须传入参数。以下是实现自动求导的主要步骤
"""
# 1）定义叶子节点及算子节点
#定义输入张量x
x=torch.Tensor([2])
"""
作用：定义输入数据 x，它是一个标量（形状为 (1,)）。

叶子节点属性：x 是用户直接创建的张量，属于叶子节点。

requires_grad：默认值为 False，因此 x.requires_grad 为 False。"""
#初始化权重参数w，偏移量b，并设置require_grad属性为True，为自动求导
w=torch.randn(1,requires_grad=True)                                 # 生成随机权重，启用梯度跟踪
b=torch.randn(1,requires_grad=True)                                 # 生成随机偏移量，启用梯度跟踪
"""
作用：定义模型的参数 w（权重）和 b（偏置）。

叶子节点属性：w 和 b 是用户直接创建的张量，属于叶子节点。

requires_grad=True：表示需要对这些张量进行梯度跟踪，PyTorch 会自动记录与它们相关的计算图，以便后续反向传播。"""
#实现正向传播
y=torch.mul(w,x)                                            #等价于w*x
z=torch.add(y,b)                                            #等价于y+b
"""
torch.mul(w, x)：

对 w 和 x 进行逐元素乘法（这里是标量乘法）。

结果 y 是中间变量，由操作产生，属于非叶子节点。

torch.add(y, b)：

将 y 和 b 相加，得到最终输出 z。

z 也是非叶子节点。"""
#查看x,w,b叶子节点的require_grad属性
print("x,w,b的require_grad属性分别是：{}，{}，{}".format(x.requires_grad,w.requires_grad,b.requires_grad))
"""
关键概念总结
        叶子节点与非叶子节点：

             叶子节点：用户直接创建的张量（如 x, w, b）。

             非叶子节点：通过操作生成的张量（如 y, z）。

             叶子节点的 grad_fn 为 None，非叶子节点的 grad_fn 指向生成它的操作。

        requires_grad=True 的作用：

             启用梯度跟踪，允许在反向传播时自动计算梯度。

             只有参数的 requires_grad 需要设置为 True（如 w, b），输入数据通常不需要。

        正向传播与计算图：

            PyTorch 会自动构建计算图（w*x → y, y+b → z）。

            计算图用于后续的自动求导（如 z.backward() 计算 w 和 b 的梯度）。
"""
#   2)查看叶子节点、非叶子节点的其他属性
#查看非叶子节点的requires_grad属性
print("y,z的require_grad属性分别是：{}，{}".format(y.requires_grad,z.requires_grad))
"""
y 和 z 是通过对 w 和 b（已启用梯度跟踪）的操作生成的，因此它们的 requires_grad 自动继承为 True。

规则：若参与计算的张量中至少有一个需要梯度，则结果张量也会需要梯度。
"""
#因与w，b有依赖关系，故y，z的requires_grad属性也是：True、True
#查看各节点是否为叶子节点
print("x,w,b,y,z是否为叶子节点：{}，{}，{}，{}，{}".format(x.is_leaf,w.is_leaf,b.is_leaf,y.is_leaf,z.is_leaf))
"""
x,w,b,y,z是否为叶子节点：True、True、True、False,False
"""
#查看叶子结点的grad_fn属性
print("x,w,b的grad_fn属性：{}，{}，{}".format(x.grad_fn,w.grad_fn,b.grad_fn))
"""
因x,w,b是用户创建的，为通过其他张量计算得到的，故x，w，b的grad_fn属性：None,None,None
叶子节点是计算图的起点，不通过任何操作生成，因此没有梯度函数（grad_fn 为 None）。

"""
#查看非叶子节点的grad_fn属性
print("y,z的grad_fn属性：{}，{}".format(y.grad_fn,z.grad_fn))
"""
y 的 grad_fn：指向生成 y 的操作 MulBackward0（乘法操作的反向传播函数）。

z 的 grad_fn：指向生成 z 的操作 AddBackward0（加法操作的反向传播函数）。

grad_fn 用于在反向传播时计算梯度。
"""
#   3）自动求导，实现梯度方向传播，即梯度的反向传播
#基于z张量进行梯度反向传播，执行backward函数之后计算图会自动清空
z.backward()
"""
作用：从张量 z 开始，沿计算图反向传播，计算所有叶子节点（w, b）的梯度。

规则：

z 必须是一个标量（单个值），否则需要指定 gradient 参数。

默认情况下，计算完成后会释放计算图以节省内存。
"""
#如果需要多次使用backward函数，需要修改参数retain_graph为True，此时梯度是累加的
#z.backward(retain_graph=True)
#查看叶子结点的梯度，x是叶子节点但他无需求导，故其梯度为None
print("参数w，b的梯度分别是：{}，{}，{}".format(w.grad,b.grad,x.grad))
#参数w,b的梯度分别为：tensor([2.]),tensor([1.]),None
"""
解释：

w.grad：z = w*x + b，对 w 的导数为 x（即 2）。

b.grad：对 b 的导数为 1。

x.grad：x 的 requires_grad=False，不需要梯度，因此为 None。
"""
#非叶子节点的梯度，执行backward函数之后，会自动清空
print("非叶子节点y，z的梯度分别为：{}，{}".format(y.grad,z.grad))
#非叶子节点y，z的梯度分别为：None，None
"""
PyTorch 默认只保留叶子节点的梯度，非叶子节点（如 y, z）的梯度在反向传播后会被自动清除。

若需保留非叶子节点的梯度，需在反向传播前调用 y.retain_grad() 和 z.retain_grad()
"""
########################################################################
"""
非标量反向传播
"""
#非标量简单示例
# x=torch.ones(2,requires_grad=True)
# y=x**2+3
# y.backward()
x=torch.ones(2,requires_grad=True)
y=x**2+3
y.sum().backward()
print(x.grad)#tensor([2.,2.])
#非标量复杂实例
#定义叶子节点张量x,形状为1*2
x=torch.tensor([[2,3]],dtype=torch.float,requires_grad=True)
#初始化雅可比矩阵
j=torch.zeros(2,2)
#初始化目标张量，形状为1*2
y=torch.zeros(1,2)
#定义y与x之间的映射关系
#y1=x1**2+3*x2，y2=x2**2+2*x1
y[0,0]=x[0,0]**2+3*x[0,1]
y[0,1]=x[0,1]**2+3*x[0,0]
#生成y1对x的梯度
y.backward(torch.Tensor([[1,0]]),retain_graph=True)
j[0]=x.grad
#梯度是累加的，故需要对x的梯度清零
x.grad=torch.zeros_like(x.grad)
"""
y.backward(torch.Tensor([[1, 0]]), retain_graph=True)：backward 方法用于进行反向传播计算梯度。torch.Tensor([[1, 0]]) 是一个权重张量，它表示只计算 y 的第一个元素对 x 的梯度。retain_graph=True 表示保留计算图，以便后续再次进行反向传播。
j[0] = x.grad：将计算得到的梯度赋值给雅可比矩阵的第一行。
x.grad = torch.zeros_like(x.grad)：由于梯度是累加的，为了计算下一个梯度，需要将 x 的梯度清零。
"""
#生成y2对x的梯度
y.backward(torch.Tensor([[0,1]]))
j[1]=x.grad
"""
y.backward(torch.Tensor([[0, 1]]))：这次使用 torch.Tensor([[0, 1]]) 作为权重张量，表示只计算 y 的第二个元素对 x 的梯度。
j[1] = x.grad：将计算得到的梯度赋值给雅可比矩阵的第二行。
"""
#显示雅可比矩阵的值
print(j)
#################################################################################################
"""
切断一些分支的反向传播
"""
x=torch.ones(2,requires_grad=True)
y=x**2+3
##对分离变量y，生成一个新变量c
c=y.detacg()
z=c*x
z.sum().backward()
# x.grad==c                               ##tensor([True,True]）
# x.grad                                  ##tensor([4.,4.])
# c.grad_fn==None                         ##True
# c.requires_grad                         ##False
###################################################################################################################
"""
使用Numpy实现机器学习任务
"""
# 1)导入需要的库

import numpy as np
from matplotlib import pyplot as plt
#  2）生成输入数据x及目标函数y。设置随机数种子，生成同一个份数据，以便用多种方法进行比较
np.random.seed(100)
x=np.linspace(-1,1,100).reshape(100,1)
print(x)
y=3*np.power(x,2)+2+0.2*np.random.rand(x.size).reshape(100,1)
print(y)
"""
功能：在数轴上从-1到1之间，均匀地打100个点，就像用尺子等距离画点一样。

例如：-1, -0.98, -0.96, ..., 0.96, 0.98, 1（共100个点）

为什么变形为(100,1)？            为了让数据形状更规范（比如后续用于机器学习模型时通常需要二维数据）。
"""
# 3）查看x，y数据分布情况
plt.scatter(x,y)
plt.show()
#  4)初始化权重参数
w1=np.random.rand(1,1)
b1=np.random.rand(1,1)
#  5)训练模型
lr=0.001#学习率
for i in range(800):
    #正向传播
    y_pred=np.power(x,2)*w1+b1
    #定义损失函数
    loss=0.5*(y_pred-y)**2
    loss=loss.sum()
    #计算梯度
    grad_w=np.sum((y_pred-y)*np.power(x,2))
    grad_b=np.sum((y_pred-y))
    #使用梯度下降法，损失值最小
    w1-=lr*grad_w
    b1-=lr*grad_b
#  6)查看可视化结果
# 绘制预测曲线（红色实线）和真实数据（蓝色散点）
plt.plot(x,y,'r-',label='predict',linewidth=4)# 使用训练后的参数计算预测值
plt.plot(x,y,color='blue',marker='o',label='true')# true data
plt.xlim(-1,1)# 设置x轴范围                      ## 真实数据点
plt.ylim(2,6)# 设置y轴范围
plt.legend() # 显示图例
plt.show()
print(w1,b1)
"""
使用Tensor及autograd实现机器学习任务
"""
# 切换后端避免显示问题
import matplotlib

matplotlib.use('Agg')  # 关键修改
torch.manual_seed(100)
dtype=torch.float
#生成x坐标数据，x为tenor，形状为100*1
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
"""
1. torch.linspace(-1, 1, 100)
作用：生成一个一维张量，包含从 -1 到 1 之间的 100 个等间隔数值。

输出示例：

tensor([-1.0000, -0.9798, -0.9596, ..., 0.9596, 0.9798, 1.0000])
形状：(100,)（长度为 100 的一维张量）。

2. torch.unsqueeze(..., dim=1)
作用：在指定维度 dim=1 处插入一个新的维度，将一维张量转换为二维张量。

输入：一维张量（形状 (100,)）。

输出：二维张量（形状 (100, 1)）。
示例：

# 原始一维张量
tensor([a1, a2, ..., a100])  
# unsqueeze(dim=1) 后变为二维张量
tensor([[a1],
        [a2],
        ...
        [a100]])
3. 最终结果 x
形状：(100, 1)
这是一个二维张量，包含 100 行、1 列的数据。
"""
#生成y坐标数据，y为tenor，形状为100*1，另加上一些噪声
y=3*x.pow(2)+2+0.2*torch.rand(x.size())
#画图，把tensor数据转换为numpy数据
# 保存散点图
plt.scatter(x.numpy(), y.numpy())
plt.savefig('scatter_plot.png')
plt.close()  # 关闭当前 figure
#随机初始化参数，参数w，b是需要学习的，故需设置requires_grad=True
w=torch.randn(1,1,dtype=dtype,requires_grad=True)
b=torch.zeros(1,1,dtype=dtype,requires_grad=True)
lr=0.001#学习率
for i in range(800):
    #forward:计算loss
    y_pred=x.pow(2).mm(w)+b
    loss=0.5*(y_pred-y)**2
    loss=loss.sum()
    #backward:自动计算梯度
    loss.backward()
    #手动更新参数，需要用torch.no_grad()更新参数
    with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

            # 因通过autograd计算的梯度会累加到grad中，所以每次循环需把梯度清零
            w.grad.zero_()
            b.grad.zero_()
plt.plot(x.numpy(),y_pred.detach().numpy(),'r-',label='predict',linewidth=4)# 使用训练后的参数计算预测值
plt.plot(x.numpy(),y.numpy(),color='blue',marker='o',label='true')# true data
plt.xlim(-1,1)# 设置x轴范围                      ## 真实数据点
plt.ylim(2,6)# 设置y轴范围
plt.legend() # 显示图例
plt.savefig('regression_plot.png')
plt.close()
print(w,b)
#########################################################################################################################
"""
使用优化器及自动微分实现机器学习任务
"""
import torch
import matplotlib.pyplot as plt
from torch import nn

# 设置随机种子以保证结果可复现
torch.manual_seed(42)

try:
    # 设置数据类型
    dtype = torch.float

    # 生成输入数据 x，在 -1 到 1 之间均匀取 100 个点，并增加一个维度
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    # 生成对应的真实标签 y，根据 y = 3 * x^2 + 2 并添加一些随机噪声
    y = 3 * x.pow(2) + 2 + 0.2 * torch.rand(x.size())


    # 初始化模型参数 w 和 b
    w = torch.randn(1, 1, dtype=dtype, requires_grad=True)
    b = torch.zeros(1, 1, dtype=dtype, requires_grad=True)

    # 定义损失函数为均方误差损失
    loss_func = nn.MSELoss()
    # 定义优化器为随机梯度下降，优化 w 和 b，学习率为 0.001
    optimizer = torch.optim.SGD([w, b], lr=0.001)

    # 训练模型
    for i in range(10000):
        # 前向传播：计算预测值 y_pred
        y_pred = x.pow(2).mm(w) + b
        # 计算损失值
        loss = loss_func(y_pred, y)

        # 反向传播：自动计算梯度
        loss.backward()

        # 更新参数
        optimizer.step()

        # 梯度清零，防止梯度累积
        optimizer.zero_grad()

        # 每 1000 次迭代打印一次损失值
        if i % 1000 == 0:
            print(f'Iteration {i}, Loss: {loss.item()}')

    # 查看可视化结果
    plt.plot(x.numpy(), y_pred.detach().numpy(), 'r-', label='predict', linewidth=4)
    plt.scatter(x.numpy(), y.numpy(), color='blue', marker='o', label='true')
    plt.xlim(-1, 1)
    plt.ylim(2, 6)
    plt.legend()
    # 保存训练结果图，避免在无图形界面环境下出错
    plt.savefig('training_result1.png')
    plt.close()

    # 打印最终的参数 w 和 b
    print(f'Final w: {w.item()}, Final b: {b.item()}')

except Exception as e:
    print(f'An error occurred: {e}')

###################################################################################################
"""
把数据集转换为带批量处理功能的迭代器
"""
# 1)构建数据迭代器
import numpy as np
#构建数据迭代器
def data_iter(features,labels,batch_size=4):
    num_examples = len(features)
    indices=list(range(num_examples))
    np.random.shuffle(indices)#样本的读取顺序是随机的
    for i in range(0,num_examples ,batch_size):
        indexs=torch.LongTensor(indices[i:min(i+batch_size,num_examples)])
        yield features.index_select(0,indexs),labels.index_select(0,indexs)
# 2）训练模型
for i in range(1000):
    for features,labels in data_iter(x,y,10):
        #forward:计算loss
        y_pred=features.pow(2).mm(w)+b
        loss=loss_func(y_pred, labels)

        #bankward:自动计算梯度
        loss.backward()

        #更新参数
        optimizer.step()
        #因通过autograd计算的梯度会累加到grad中，所以每次循环需要把梯度清零
        optimizer.zero_grad()
y_p=x.pow(2).mm(w).detach().numpy()+b.detach().numpy()
plt.plot(x.numpy(), y_pred.detach().numpy(), 'r-', label='predict', linewidth=4)
plt.scatter(x.numpy(), y.numpy(), color='blue', marker='o', label='true')
plt.xlim(-1, 1)
plt.ylim(2, 6)
plt.legend()
# 保存训练结果图，避免在无图形界面环境下出错
plt.savefig('training_result2.png')
plt.close()
# 打印最终的参数 w 和 b
print(f'Final w: {w.item()}, Final b: {b.item()}')
##########################################################################################################
"""
使用TensorFlow2实现机器学习任务
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# 生成训练数据
np.random.seed(100)
x = np.linspace(-1, 1, 100).reshape(100, 1).astype(np.float32)
y = 3 * np.power(x, 2) + 2 + 0.2 * np.random.rand(x.size).reshape(100, 1).astype(np.float32)

# 创建权重变量 w 和 b，并用随机值初始化
# TensorFlow 的变量在整个计算图保留其值
w = tf.Variable(tf.random.uniform([1], 0, 1.0, dtype=tf.float32))
b = tf.Variable(tf.zeros([1], dtype=tf.float32))


# 构建模型
# 定义模型
class CustNet:
    # 正向传播
    def __call__(self, x):
        return tf.pow(x, 2) * w + b

    # 损失函数
    def loss_func(self, y_true, y_pred):
        return tf.reduce_mean((y_true - y_pred) ** 2 / 2)


model = CustNet()

# 训练模型
epochs = 14000
for epoch in tf.range(1, epochs):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = model.loss_func(y, predictions)
    # 反向传播求梯度
    dw, db = tape.gradient(loss, [w, b])
    # 梯度下降法更新参数
    w.assign(w - 0.001 * dw)
    b.assign(b - 0.001 * db)

# 可视化结果
plt.figure()
plt.scatter(x, y, color='blue', marker='o', label='true')
plt.plot(x, b.numpy() + w.numpy() * x ** 2, 'r-', label='predict', linewidth=4)
plt.legend()
plt.savefig('training_result3.png')
plt.close()
print(f'Final w: {w.item()}, Final b: {b.item()}')



