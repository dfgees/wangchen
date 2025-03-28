"""
PyTorch神经网络工具箱
1.神经网络核心组件
2.构建神经网络的主要工具
3.构建模型
4.训练模型
5.实现神经网络实例
"""
from torch.utils.tensorboard import SummaryWriter

"""
神经网络核心组件
"""
#层：神经网络的基本结构，将输入张量转化为输出张量
#模型：由层构成的网络
#损失函数：参数学习的目标函数，通过最小化损失函数来学习各种参数
#优化器：如在使损失值最小时，就会涉及优化器
"""
构建神经网络的主要工具
一、nn.Module 的核心作用：
     nn.Module 是 PyTorch 中所有神经网络模型的基类，它提供了 模块化构建模型的基础框架，主要作用包括：

1.参数管理:

     自动跟踪所有子模块（如 nn.Linear, nn.Conv2d 等）的可训练参数（weight 和 bias）。

     通过 .parameters() 方法获取所有参数，方便传递给优化器（如 optim.SGD）。

2.自动微分支持:

     在 forward() 中定义的计算流程，会自动构建计算图，支持反向传播（通过 backward()）。

3.模块化结构:

     允许嵌套子模块（如 nn.Sequential），实现复杂模型的层次化组织。

4.支持模型复用（例如将预训练模型作为子模块）。

5.设备移动（CPU/GPU）:

      通过 .to(device) 一键将模型参数和计算迁移到 GPU 或 CPU。

6.模型保存与加载:

      通过 .state_dict() 保存模型参数，通过 .load_state_dict() 加载参数。


二、nn.functional 的核心作用
1.无状态操作：

     提供无需维护参数的函数（如激活函数、损失函数、卷积/池化操作等）。

     适合动态计算或需要临时使用的操作。

2.灵活性：

     允许在模型前向传播（forward）中直接调用函数，支持条件分支或复杂逻辑。

3.覆盖常见操作：

     包含激活函数（relu, sigmoid）、损失函数（cross_entropy, mse_loss）、卷积（conv2d）、归一化（batch_norm）、池化（max_pool2d）等。
     
     
三、nn.functional vs nn.Module
特性	                  nn.Module (如 nn.ReLU)	                          nn.functional (如 F.relu)

参数管理	              自动跟踪参数（如权重、偏置）	                          无参数，需手动传递参数（若有）
适用场景	              定义模型中的固定层（如全连接层）	                     动态操作（如条件激活、自定义损失）
状态管理	        自动处理状态（如 Dropout 的 training 模式）	                 需手动控制（如设置 training=True）
代码风格	                       面向对象	                                           函数式编程
"""
#使用PyTorch构建模型的方法大致有三种
#1）继承nn.Module基类构建模型
#2）使用nn.Sequential按层顺序构建模型
#3）继承nn.Module基类构建模型，再使用相关模型容器进行封装
"""
继承nn.Module基类构建模型
"""
#1)导入模块
import torch
from torch import nn
import torch.nn.functional as F


#2）构建模型
class Model_Seq(nn.Module):  #定义了一个名为 Model_Seq 的类，它继承自 nn.Module，这是 PyTorch 中所有神经网络模块的基类。
    """
    通过继承基类nn.Module来构建模型
    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        """
        初始化方法，接收四个参数：
            in_dim：输入层的维度。
            n_hidden_1：第一个隐藏层的神经元数量。
            n_hidden_2：第二个隐藏层的神经元数量。
            out_dim：输出层的维度。
        """

        #初始化方法，接收四个参数：
        super(Model_Seq, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_dim, n_hidden_1)  #
        self.bn1 = nn.BatchNorm1d(n_hidden_1)
        self.linear2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.bn2 = nn.BatchNorm1d(n_hidden_2)
        self.out = nn.Linear(n_hidden_2, out_dim)

    """
super(Model_Seq, self).__init__()：调用父类 nn.Module 的初始化方法，确保正确初始化。
self.flatten = nn.Flatten()：创建一个 nn.Flatten 层，用于将输入数据展平为一维向量。
self.linear1 = nn.Linear(in_dim, n_hidden_1)：创建第一个全连接层（线性层），输入维度为 in_dim，输出维度为 n_hidden_1。
self.bn1 = nn.BatchNorm1d(n_hidden_1)：创建第一个批量归一化层（BatchNorm1d），用于对第一个全连接层的输出进行归一化处理，加速模型收敛。
self.linear2 = nn.Linear(n_hidden_1, n_hidden_2)：创建第二个全连接层，输入维度为 n_hidden_1，输出维度为 n_hidden_2。
self.bn2 = nn.BatchNorm1d(n_hidden_2)：创建第二个批量归一化层，用于对第二个全连接层的输出进行归一化处理。
self.out = nn.Linear(n_hidden_2, out_dim)：创建输出层，输入维度为 n_hidden_2，输出维度为 out_dim。
"""

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.softmax(x, dim=1)
        return x


"""
def forward(self, x)：定义前向传播方法，用于定义数据在模型中的流动过程。
x = self.flatten(x)：将输入数据 x 展平为一维向量。
x = self.linear1(x)：将展平后的数据输入到第一个全连接层。
x = self.bn1(x)：对第一个全连接层的输出进行批量归一化处理。
x = F.relu(x)：对归一化后的输出应用 ReLU 激活函数，引入非线性。
x = self.linear2(x)：将经过第一个隐藏层处理后的数据输入到第二个全连接层。
x = self.bn2(x)：对第二个全连接层的输出进行批量归一化处理。
x = F.relu(x)：对归一化后的输出再次应用 ReLU 激活函数。
x = self.out(x)：将经过第二个隐藏层处理后的数据输入到输出层。
x = F.softmax(x, dim=1)：对输出层的输出应用 Softmax 激活函数，将输出转换为概率分布，dim=1 表示在第二个维度（即类别维度）上进行 Softmax 操作。
return x：返回最终的输出结果。"""

#3)查看模型
##对一些超参数赋值
in_dim, n_hidden_1, n_hidden_2, out_dim = 28 * 28, 300, 100, 10
model_seq = Model_Seq(in_dim, n_hidden_1, n_hidden_2, out_dim)
print(model_seq)
"""
in_dim, n_hidden_1, n_hidden_2, out_dim = 28 * 28, 300, 100, 10：对模型的超参数进行赋值，这里假设输入数据是 28x28 的图像，因此输入维度为 28 * 28，第一个隐藏层有 300 个神经元，第二个隐藏层有 100 个神经元，输出层有 10 个神经元，通常用于处理 10 分类问题。
model_seq = Model_Seq(in_dim, n_hidden_1, n_hidden_2, out_dim)：创建 Model_Seq 类的一个实例，传入超参数。
print(model_seq)：打印模型的结构，方便查看模型的组成部分。
"""
"""
输出为：
Model_Seq(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear1): Linear(in_features=784, out_features=300, bias=True)
  (bn1): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (linear2): Linear(in_features=300, out_features=100, bias=True)
  (bn2): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (out): Linear(in_features=100, out_features=10, bias=True)
)
in_features=784：输入特征的数量，这里是 784，通常对应于 28x28 图像展平后的维度（28 * 28 = 784）。
out_features=300：输出特征的数量，即该层的神经元数量为 300。
bias=True：表示该层使用偏置项。
300：输入特征的数量，与 linear1 层的输出特征数量一致。
eps=1e-05：一个小的常数，用于数值稳定性，防止分母为零。
momentum=0.1：用于计算移动平均和移动方差的动量值，控制历史统计信息对当前统计信息的影响程度。
affine=True：表示该层具有可学习的缩放因子和偏移量。
track_running_stats=True：表示在训练过程中跟踪输入数据的均值和方差。
300：输入特征的数量，与 linear1 层的输出特征数量一致。
eps=1e-05：一个小的常数，用于数值稳定性，防止分母为零。
momentum=0.1：用于计算移动平均和移动方差的动量值，控制历史统计信息对当前统计信息的影响程度。
affine=True：表示该层具有可学习的缩放因子和偏移量。
track_running_stats=True：表示在训练过程中跟踪输入数据的均值和方差。 
in_features=300：输入特征的数量，与 bn1 层的输出特征数量一致。
out_features=100：输出特征的数量，即该层的神经元数量为 100。
bias=True：表示该层使用偏置项。
"""
##############################################################################################
"""
使用nn.Sequential按层顺序构建模型
"""
#                  1.利用可变参量
#1）导入模块
import torch
from torch import nn

#2)构建模型
seq_arg = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_dim, n_hidden_1),
    nn.BatchNorm1d(n_hidden_1),
    nn.ReLU(),
    nn.Linear(n_hidden_1, n_hidden_2),
    nn.BatchNorm1d(n_hidden_2),
    nn.ReLU(),
    nn.Linear(n_hidden_2, out_dim),
    nn.Softmax(dim=1),
)
"""
       nn.Sequential：这是一个有序的容器，它按照传入的顺序依次将各个层组合在一起，形成一个神经网络模型。当输入数据传入这个模型时，数据会依次通过这些层进行处理。

       nn.Flatten()：该层的作用是将输入的多维张量展平为一维张量。在处理图像等多维数据时，全连接层通常要求输入是一维向量，因此使用 Flatten 层将输入数据进行展平。例如，如果输入是一个形状为 (batch_size, channels, height, width) 的图像张量，经过 Flatten 层后会变成形状为 (batch_size, channels * height * width) 的一维张量。
       nn.Linear(in_dim, n_hidden_1)：这是一个全连接层（线性层），它对输入进行线性变换。具体来说，它会将输入的 in_dim 维向量通过矩阵乘法和偏置加法转换为 n_hidden_1 维向量。数学公式为 
，其中x是输入向量,W是权重矩阵,b是偏置向量。
       nn.BatchNorm1d(n_hidden_1)：一维批量归一化层，用于对输入数据进行归一化处理。它可以加速模型的收敛速度，提高模型的稳定性。
该层会对输入的每个特征维度进行归一化，使得数据的均值接近 0，方差接近 1。
       nn.ReLU()：ReLU（Rectified Linear Unit）激活函数，它的作用是引入非线性。ReLU 函数的定义为 
f(x)=max(0,x)，即当输入 大于 0 时，输出为x；当输入x小于等于 0 时，输出为 0。通过引入非线性，模型可以学习到更复杂的模式。
       nn.Linear(n_hidden_1, n_hidden_2)：另一个全连接层，将 n_hidden_1 维的输入向量转换为 n_hidden_2 维的输出向量。
       nn.BatchNorm1d(n_hidden_2)：对 n_hidden_2 维的输入数据进行批量归一化处理。
       nn.ReLU()：再次应用 ReLU 激活函数，引入非线性。
       nn.Linear(n_hidden_2, out_dim)：输出层的全连接层，将 n_hidden_2 维的输入向量转换为 out_dim 维的输出向量。
通常，out_dim 对应于分类问题的类别数量。
       nn.Softmax(dim=1)：Softmax 激活函数，用于将输出转换为概率分布。dim=1 表示在第二个维度（即类别维度）上进行 Softmax 操作。
"""
#3)查看模型
in_dim, n_hidden_1, n_hidden_2, out_dim = 28 * 28, 300, 100, 10
print(seq_arg)
##############################################################################################################
#                          2)使用add_module方法
#1)构建模型
Seq_module = nn.Sequential()
"""
nn.Sequential() 是 PyTorch 中用于构建顺序模型的容器类。它允许你按顺序堆叠多个神经网络层，形成一个完整的模型。这里创建了一个空的 nn.Sequential 对象 Seq_module，后续会向这个容器中添加不同的神经网络层。
"""
Seq_module.add_module('flatten', nn.Flatten())
Seq_module.add_module('linear1', nn.Linear(in_dim, n_hidden_1))
Seq_module.add_module('bn1', nn.BatchNorm1d(n_hidden_1))
Seq_module.add_module("rule1", nn.ReLU())
Seq_module.add_module('linear2', nn.Linear(n_hidden_1, n_hidden_2))
Seq_module.add_module('bn2', nn.BatchNorm1d(n_hidden_2))
Seq_module.add_module("rule2", nn.ReLU())
Seq_module.add_module("out", nn.Linear(n_hidden_2, out_dim))
Seq_module.add_module('softmax', nn.Softmax(dim=1))
"""
add_module 是 nn.Sequential 类的一个方法，用于向容器中添加一个具有指定名称的模块（即神经网络层）
"""
in_dim, n_hidden_1, n_hidden_2, out_dim = 28 * 28, 300, 100, 10

print(Seq_module)
#################################################################################################################
#                            3.使用OrderDict方法
#1）导入模块
import torch
from torch import nn
from collections import OrderedDict

Seq_dict = nn.Sequential(OrderedDict([
    ('flatten', nn.Flatten()),
    ('linear1', nn.Linear(in_dim, n_hidden_1)),
    ('bn1', nn.BatchNorm1d(n_hidden_1)),
    ('rule1', nn.ReLU()),
    ('linear2', nn.Linear(n_hidden_1, n_hidden_2)),
    ('bn2', nn.BatchNorm1d(n_hidden_2)),
    ('rule2', nn.ReLU()),
    ('out', nn.Linear(n_hidden_2, out_dim)),
    ("softmax", nn.Softmax(dim=1))]))
n_dim, n_hidden_1, n_hidden_2, out_dim = 28 * 28, 300, 100, 10

print(Seq_dict)
"""
继承nn.Module基类并应用模型容器来构建模型
"""
import torch
from torch import nn
import torch.nn.functional as F


#2)构建模型
class Model_lay(nn.Module):
    """
    使用nn.Sequential构建网络，Sequential()函数的功能是将网络的层组合到一起
    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Model_lay, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.out = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self, self.layer1(x))
        x = F.relu(self, self.layer2(x))
        x = F.softmax(self.out(x), dim=1)
        return x


in_dim, n_hidden_1, n_hidden_2, out_dim = 28 * 28, 300, 100, 10
model_lay = Model_lay(in_dim, n_hidden_1, n_hidden_2, out_dim)
print(model_lay)
#2.使用nn.MolduleList模型容器
#1）导入模块
import torch
from torch import nn
import torch.nn.functional as F


#2）构建模型
class Model_lst(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Model_lst, self).__init__()
        self.layers = nn.ModuleList([
            nn.Flatten(),
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1),
            nn.ReLU(),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2),
            nn.ReLU(),
            nn.Linear(n_hidden_2, out_dim),
            nn.Softmax(dim=1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


in_dim, n_hidden_1, n_hidden_2, out_dim = 28 * 28, 300, 100, 10
model_lst = Model_lst(in_dim, n_hidden_1, n_hidden_2, out_dim)
print(model_lst)
#                             使用nn.ModuleDict模型容器
#1)导入模块
import torch
from torch import nn


#2）构建模型
class Model_dict(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Model_dict, self).__init__()
        self.layers_dict = nn.ModuleDict({
            'flatten': nn.Flatten(),
            'linear1': nn.Linear(in_dim, n_hidden_1),
            'bn1': nn.BatchNorm1d(n_hidden_1),
            'rule': nn.ReLU(),
            'linear2': nn.Linear(n_hidden_1, n_hidden_2),
            'bn2': nn.BatchNorm1d(n_hidden_2),
            'out': nn.Linear(n_hidden_2, out_dim),
            'softmax': nn.Softmax(dim=1)
        })

    def forward(self, x):
        layers = ['flatten', 'linear1', 'bn1', 'rule', 'linear2', 'bn2', 'out', 'softmax']
        for layer in layers:
            x = self.layers_dict[layer](x)

        return x

    """
    def forward(self, x)：定义前向传播方法，该方法描述了数据在模型中的流动过程。输入参数 x 是输入数据。
    layers = ['flatten', 'linear1', 'bn1', 'rule', 'linear2', 'bn2', 'out', 'softmax']：定义了一个列表，包含了模型中各层的名称，用于指定数据流动的顺序。
    for layer in layers: x = self.layers_dict[layer](x)：通过循环遍历 layers 列表，依次将输入数据 x 传入对应的层进行处理，并将处理后的结果更新为新的 x。
    return x：返回最终的输出结果。"""


in_dim, n_hidden_1, n_hidden_2, out_dim = 28 * 28, 300, 100, 10
model_dict = Model_dict(in_dim, n_hidden_1, n_hidden_2, out_dim)
print(model_dict)

"""
自定义网络模块
"""
#定义图3-4a所示的残差模块
import torch
from torch import nn
from torch.nn import functional as F


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        """
        conv1：第一个卷积层
            ◦ 输入通道数 in_channels
            ◦ 输出通道数 out_channels
            ◦ 卷积核大小 3x3
            ◦ 步长由参数 stride 指定
            ◦ 填充 1（保持特征图尺寸不变*，当 stride=1 时）
        
        bn1：批量归一化层
              对 conv1 的输出进行归一化，加速训练

        conv2：第二个卷积层
             输入和输出通道数均为 out_channels
             其他参数与 conv1 相同（包括相同的 stride）
             bn2：第二个批量归一化层
             对 conv2 的输出进行归一化
             """

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        output = F.relu(x + output)
        """
        计算流程:
          第一次卷积
               x → conv1 → bn1 → ReLU激活
               输入特征图经过卷积、归一化和激活函数

          第二次卷积
               → conv2 → bn2
               继续卷积和归一化，但没有激活函数

          残差连接
               → x + output → ReLU激活
               将原始输入 x 与卷积结果 output 相加
               对相加结果进行 ReLU 激活
        """


"""
        类定义：RestNetBasicBlock 继承自 nn.Module，是残差网络的基本单元。

        初始化方法：

            输入参数：in_channels（输入通道数）、out_channels（输出通道数）、stride（卷积步长）。

            定义两个卷积层 conv1 和 conv2，每个卷积层后接批量归一化层 bn1 和 bn2。

        前向传播：

            输入 x 依次经过 conv1 → bn1 → ReLU → conv2 → bn2。

            将原始输入 x 与 output 相加，再通过 ReLU 激活。
            
 残差连接的设计意图:
(1) 核心思想
跳跃连接（Skip Connection）：让输入 x 直接绕过部分网络层，与深层输出相加。

目的：缓解深层网络的梯度消失问题，使网络更容易训练。

(2) 数学表示
理想情况：输出 = F(x) + x

其中 F(x) 是卷积层和归一化的组合
"""
#定义图3-4b所示的残差模块
"""
这段代码定义了一个用于 ResNet（残差网络）的基本残差块类 ResNetBasicBlock，它继承自 nn.Module。ResNet 是一种深度卷积神经网络，引入了残差连接来解决深度神经网络训练过程中的梯度消失和梯度爆炸问题，使得网络可以训练更深的层次。
下面我们逐行详细解释这段代码："""


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBasicBlock, self).__init__()
        """
class ResNetBasicBlock(nn.Module)：定义了一个名为 ResNetBasicBlock 的类，它继承自 nn.Module，这是 PyTorch 中所有神经网络模块的基类。
def __init__(self, in_channels, out_channels, stride)：类的初始化方法，接收三个参数：
in_channels：输入特征图的通道数。
out_channels：输出特征图的通道数。
stride：一个包含两个元素的列表或元组，分别表示第一个卷积层和第二个卷积层的步长。
super(ResNetBasicBlock, self).__init__()：调用父类 nn.Module 的初始化方法，确保正确初始化父类的属性和方法。
        """
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        """       
self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)：定义第一个二维卷积层。
in_channels：输入特征图的通道数。
out_channels：输出特征图的通道数。
kernel_size=3：卷积核的大小为 3x3。
stride=stride[0]：卷积的步长，使用 stride 列表的第一个元素。
padding=1：在输入特征图的边界填充 1 个像素，以保持特征图的空间尺寸不变（当步长为 1 时）。

self.bn1 = nn.BatchNorm2d(out_channels)：定义第一个二维批量归一化层，用于对第一个卷积层的输出进行归一化处理，加速模型收敛。

self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)：定义第二个二维卷积层，输入和输出通道数相同。
stride=stride[1]：卷积的步长，使用 stride 列表的第二个元素。

self.bn2 = nn.BatchNorm2d(out_channels)：定义第二个二维批量归一化层，用于对第二个卷积层的输出进行归一化处理。
        """
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels),
        )

    # """
    # self.extra：定义一个 nn.Sequential 容器，用于实现残差连接。当输入特征图的通道数或空间尺寸与输出特征图不一致时，需要对输入进行额外的变换，使其能够与卷积层的输出进行相加。
    # nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0)：一个 1x1 卷积层，用于调整输入特征图的通道数和空间尺寸，使其与卷积层的输出相匹配。
    # nn.BatchNorm2d(out_channels)：对 1x1 卷积层的输出进行批量归一化处理。
    # """
    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        output = self.bn2(out)
        assert isinstance(out)
        return F.relu(extra_x + out)


#  """
# def forward(self, x)：定义前向传播方法，描述数据在模型中的流动过程。输入参数 x 是输入特征图。
# extra_x = self.extra(x)：对输入特征图 x 进行额外的变换，得到 extra_x，用于残差连接。
# output = self.conv1(x)：将输入特征图 x 传入第一个卷积层 self.conv1 进行卷积操作。
# out = F.relu(self.bn1(output))：对第一个卷积层的输出进行批量归一化处理，然后通过 ReLU 激活函数引入非线性。
# out = self.conv2(out)：将经过第一个卷积层和激活函数处理后的特征图传入第二个卷积层 self.conv2 进行卷积操作。
# output = self.bn2(out)：对第二个卷积层的输出进行批量归一化处理。
# return F.relu(extra_x + out)：将经过额外变换的输入 extra_x 与第二个卷积层的输出 out 相加，然后通过 ReLU 激活函数，得到最终的输出。
# """

#组合这两个模块得到现代经典的RestNet18网络结构
class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        """
class RestNet18(nn.Module)：定义了一个名为 RestNet18 的类，它继承自 nn.Module，这是 PyTorch 中所有神经网络模块的基类，继承它可以方便地使用 PyTorch 提供的自动求导、参数管理等功能。
def __init__(self)：类的初始化方法，用于初始化网络的各个层。
super(RestNet18, self).__init__()：调用父类 nn.Module 的初始化方法，确保父类的属性和方法被正确初始化。
"""
        self.cunv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        """
self.cunv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)：定义第一个卷积层。
3：输入通道数，通常对应 RGB 图像的三个通道。
64：输出通道数，意味着该卷积层会输出 64 个特征图。
kernel_size=7：卷积核的大小为 7x7。
stride=2：卷积的步长为 2，会使特征图的尺寸缩小。
padding=3：在输入特征图的边界填充 3 个像素，以控制输出特征图的尺寸。
"""
        self.bn1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        """
self.bn1 = nn.BatchNorm2d(64)：定义二维批量归一化层，对卷积层的输出进行归一化处理，加速模型收敛，这里的 64 是输入特征图的通道数。
self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)：定义最大池化层。
kernel_size=3：池化核的大小为 3x3。
stride=2：池化的步长为 2，进一步缩小特征图的尺寸。
padding=1：在输入特征图的边界填充 1 个像素。
"""

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetBasicBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetBasicBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetBasicBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))
        """
self.layer1、self.layer2、self.layer3 和 self.layer4：分别定义了四个残差块层，每个层由两个 RestNetBasicBlock 残差块组成。
RestNetBasicBlock 是之前定义的残差块类，通过残差连接可以缓解梯度消失和梯度爆炸问题，使得网络可以学习到更深层次的特征。
例如，self.layer1 中的残差块输入和输出通道数都是 64，步长为 1，不会改变特征图的通道数和尺寸；
而 self.layer2 中的第一个残差块输入通道数为 64，输出通道数为 128，步长为 [2, 1]，会使特征图的通道数翻倍，尺寸缩小。
"""

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        """
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))：
        定义自适应全局平均池化层，将任意尺寸的输入特征图池化为 1x1 的特征图，这样可以将不同尺寸的输入统一到相同的维度，方便后续全连接层处理。
        """

        self.fc = nn.Linear(512, 10)
        """
        self.fc = nn.Linear(512, 10)：
        定义全连接层，将全局平均池化层输出的 512 维特征向量映射到 10 维输出，通常用于 10 分类任务。
        """

    def forward(self, x):
        out = self.cunv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out


"""
def forward(self, x)：定义前向传播方法，描述数据在网络中的流动过程。输入参数 x 是输入的图像数据。
out = self.cunv1(x)：将输入数据传入第一个卷积层进行卷积操作。
out = self.layer1(out)、out = self.layer2(out)、out = self.layer3(out)、out = self.layer4(out)：
依次将数据传入四个残差块层进行特征提取。
out = self.avgpool(out)：将残差块层的输出传入全局平均池化层进行池化操作。
out = out.view(x.size(0), -1)：这里原代码有误，应该是 out = out.view(x.size(0), -1)，将全局平均池化层输出的特征图展平为一维向量，x.size(0) 表示批量大小，-1 表示自动计算剩余的维度。
out = self.fc(out)：将展平后的一维向量传入全连接层进行分类，得到最终的输出。
return out：返回最终的分类结果。
"""

#训练模型
"""
构建模型后，接下来就是训练模型。PyTorch训练模型主要包括加载数据集，损失计算，定义优化算法，反向传播，参数更新等主要步骤。
1）加载和预处理数据集
2）定义损失函数
3）定义优化方法
4）循环训练模型
设置为训练模式：
model.train()
调用model.train()会把所有的module设置为训练模式
梯度清零：
optimizer.zero_grad()
在默认的基础下梯度是累加的，需要手工把梯度初始化或清零，调用optimizer.zero_grad()即可。
求损失值：
y_prey=model(x)
loss=loss_fun(y_prey,y_true)
自动求导，实现梯度的反向传播：
loss.backward()
更新参数：
optimizer.step()
5)循环测试或验证模型。
设置为测试或验证模式：
model.eval()
调用model.eval()会把所有的training属性设置为False.
在不跟踪梯度的模式下计算损失值，预测值等：
with.torch.no_grad()
6)可视化结果
"""
#实现神经网络实例
#准备数据
#1）导入必要的模块
import numpy as np
import torch
#导入PyTorch内置的MNIST数据
from torchvision.datasets import mnist
#导入预处理模块
import torchvision.transforms as transforms
import torch.utils.data as DataLoader
#导入nn及优化器
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
"""
numpy：一个强大的数值计算库，在处理数据时经常会用到，比如进行矩阵运算等。
torch：PyTorch 的核心库，提供了张量操作、自动求导等功能。
torchvision.datasets.mnist：用于下载和加载 MNIST 数据集，MNIST 是一个经典的手写数字图像数据集，包含 60000 张训练图像和 10000 张测试图像。
torchvision.transforms：用于对图像数据进行预处理，例如将图像转换为张量、归一化等。
torch.utils.data.DataLoader：用于创建数据加载器，方便批量加载数据。
torch.nn.functional：包含了许多常用的神经网络函数，如激活函数、损失函数等。
torch.optim：提供了各种优化算法，如随机梯度下降（SGD）、Adam 等。
torch.nn：用于构建神经网络的模块，如全连接层、卷积层等。
"""

#2)定义一些超参数
train_batch_size = 64
test_batch_size = 128
learning_rate = 0.01
num_epoch = 20
lr = 0.01
momentum = 0.5
"""
train_batch_size：训练时每个批次包含的样本数量，设置为 64 表示每次从训练数据中取出 64 个样本进行训练。
test_batch_size：测试时每个批次包含的样本数量，设置为 128 表示每次从测试数据中取出 128 个样本进行测试。
learning_rate 和 lr：学习率，控制模型参数更新的步长。学习率过大可能导致模型无法收敛，学习率过小则会使训练速度变慢。这里设置为 0.01。
num_epoch：训练的轮数，即整个训练数据集被模型学习的次数，设置为 20 表示模型会对整个训练数据集学习 20 次。
momentum：动量，用于加速 SGD 优化算法的收敛速度，设置为 0.5。
"""
#3)下载数据并对数据进行预处理
#定义预处理函数
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0, 5], [0, 5])])
#下载数据，并对数据进行预处理
train_dataset = mnist.MNIST('./data', train=True, transform=transform, download=False)
test_dataset = mnist.MNIST('./data', train=False, transform=transform, )
#得到一个生成器
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=test_batch_size,shuffle=False)
"""
定义预处理函数：
transforms.Compose：用于将多个预处理操作组合在一起。
transforms.ToTensor()：将 PIL 图像或 NumPy 数组转换为 PyTorch 张量。
transforms.Normalize([0, 5], [0, 5])：对图像数据进行归一化处理，将图像的像素值归一化到指定的均值和标准差范围内。不过这里的参数 [0, 5] 可能有误，通常 MNIST 数据集的归一化参数是 [0.1307] 和 [0.3081]。
下载数据并进行预处理：
mnist.MNIST：用于下载和加载 MNIST 数据集。
'./data'：指定数据的存储路径。
train=True 表示加载训练数据集，train=False 表示加载测试数据集。
transform=transform：对数据应用前面定义的预处理函数。
download=False：表示不下载数据，如果数据已经下载过，则直接使用。
创建数据加载器：
DataLoader：用于创建数据加载器，方便批量加载数据。
batch_size：指定每个批次包含的样本数量。
shuffle=True 表示在每个 epoch 开始时打乱训练数据的顺序，有助于提高模型的泛化能力；shuffle=False 表示不打乱测试数据的顺序。
"""
#可视化数据源
import matplotlib.pyplot as plt
"""
matplotlib.pyplot 是 Python 中一个常用的绘图库，plt 是其常用的别名。这里使用它来绘制图像，以可视化展示 MNIST 数据集中的手写数字图像。
"""
examples=enumerate(test_loader)
batch_idx,(example_data,example_targets)=next(examples)
"""
enumerate(test_loader)：enumerate 函数会为可迭代对象 test_loader 中的每个元素添加一个索引。test_loader 是之前创建的数据加载器，用于批量加载测试数据。通过 enumerate 函数，我们可以在遍历 test_loader 时同时获取每个批次的索引和数据。
next(examples)：next 函数用于从迭代器 examples 中取出下一个元素。这里取出的是第一个批次的数据，返回一个元组 (batch_idx, (example_data, example_targets))。
batch_idx：当前批次的索引。
example_data：当前批次的图像数据，是一个四维张量，形状通常为 (batch_size, channels, height, width)，对于 MNIST 数据集，channels 为 1（因为是灰度图像），height 和 width 均为 28。
example_targets：当前批次的真实标签，是一个一维张量，形状为 (batch_size,)。
"""
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0],cmap='gray',interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
"""
for i in range(6)：使用 for 循环遍历 6 次，每次绘制一张图像。
plt.subplot(2, 3, i + 1)：创建一个 2 行 3 列的子图布局，并选择当前要绘制的子图位置。i + 1 表示当前子图的编号，从左到右、从上到下依次递增。
plt.tight_layout()：自动调整子图参数，使子图之间的间距合适，避免图像重叠。
plt.imshow(example_data[i][0], cmap='gray', interpolation='none')：
example_data[i][0]：取出当前批次中第 i 张图像的数据，由于图像是灰度图像，[0] 用于获取单通道的图像数据。
cmap='gray'：指定使用灰度颜色映射来显示图像。
interpolation='none'：指定不使用插值算法，直接显示原始像素值。
plt.title("Ground Truth: %d".format(example_targets[i]))：为当前子图设置标题，显示该图像的真实标签。不过这里代码存在一个小错误，"Ground Truth: %d" 是旧的字符串格式化方式，应该使用 f"Ground Truth: {example_targets[i]}" 或者 "Ground Truth: {}".format(example_targets[i])。
plt.xticks([]) 和 plt.yticks([])：分别用于隐藏当前子图的 x 轴和 y 轴刻度，使图像更加简洁。   
"""

#1)构建网络
"""
这段代码主要完成了两个关键任务：一是定义一个简单的全连接神经网络模型，二是实例化这个模型，同时配置损失函数和优化器。下面为你详细讲解代码各部分的功能和含义。
"""
class Net(nn.Module):
    """
    使用nn.Sequential来构建网络，sequential()函数的功能是将网络的层组合到一起
    """
    def __init__(self,in_dim,n_hidden1,n_hidden2,out_dim):
        super(Net,self).__init__()
        self.flatten=nn.Flatten()
        self.layer1=nn.Sequential(nn.Linear(in_dim,n_hidden1),nn.BatchNorm1d(n_hidden1))
        self.layer2=nn.Sequential(nn.Linear(n_hidden1,n_hidden2),nn.BatchNorm1d(n_hidden2))
        self.out=nn.Sequential(nn.Linear(n_hidden2,out_dim))

    def forward(self,x):
        x=self.flatten(x)
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.softmax(self.out(x),dim=1)
        return x
"""
super(Net, self).__init__()：调用父类 nn.Module 的初始化方法。
self.flatten = nn.Flatten()：定义一个 Flatten 层，用于将输入的多维张量展平为一维向量，方便后续全连接层处理。
self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden1), nn.BatchNorm1d(n_hidden1))：nn.Sequential 用于将多个层按顺序组合成一个模块。这里组合了一个全连接层 nn.Linear(in_dim, n_hidden1) 和一个批量归一化层 nn.BatchNorm1d(n_hidden1)。全连接层将输入维度 in_dim 映射到隐藏层维度 n_hidden1，批量归一化层用于加速网络收敛和提高模型的稳定性。
self.layer2 = nn.Sequential(nn.Linear(n_hidden1, n_hidden2), nn.BatchNorm1d(n_hidden2))：类似 layer1，将 n_hidden1 维度的输入映射到 n_hidden2 维度，并进行批量归一化。
self.out = nn.Sequential(nn.Linear(n_hidden2, out_dim))：最后一个全连接层，将 n_hidden2 维度的输入映射到输出维度 out_dim。
"""
"""
def forward(self, x)：定义了网络的前向传播过程，即输入数据 x 如何通过网络得到输出。
x = self.flatten(x)：将输入数据 x 展平为一维向量。
x = F.relu(self.layer1(x))：将展平后的数据输入到 layer1 模块，然后通过 ReLU 激活函数引入非线性。
x = F.relu(self.layer2(x))：将 layer1 的输出输入到 layer2 模块，同样经过 ReLU 激活函数。
x = F.softmax(self.out(x), dim=1)：将 layer2 的输出输入到 out 模块，最后通过 Softmax 函数将输出转换为概率分布，dim=1 表示在第 1 个维度（即类别维度）上进行 Softmax 操作。
"""
#2）实例化网络
#检测是否有可用的GPU,有则使用，否则使用CPU
device=torch.device("cude:0"if torch.cuda.is_available() else "cpu")
#实例化网络
model=Net(28*28,300,100,10)
model.to(device)
#定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=lr,momentum=momentum)
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")：检测当前环境中是否有可用的 GPU，如果有则使用第一个 GPU（cuda:0），否则使用 CPU。
model = Net(28 * 28, 300, 100, 10)：实例化 Net 类，创建一个神经网络模型。输入维度 in_dim 为 28 * 28，因为 MNIST 图像的尺寸是 28x28，将其展平后得到 784 维向量；第一个隐藏层维度 n_hidden1 为 300，第二个隐藏层维度 n_hidden2 为 100，输出维度 out_dim 为 10，因为 MNIST 数据集有 10 个类别（数字 0 - 9）。
model.to(device)：将模型移动到指定的设备（GPU 或 CPU）上进行计算。
criterion = nn.CrossEntropyLoss()：定义损失函数为交叉熵损失函数，常用于多分类问题。
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)：定义优化器为随机梯度下降（SGD）优化器，用于更新模型的参数。model.parameters() 表示需要优化的模型参数，lr 是学习率，momentum 是动量，用于加速收敛。
"""

#训练模型
losses=[]
acces=[]
eval_losses=[]
eval_acces=[]
writer=SummaryWriter(log_dir='./logs',comment='train-loss')
"""
losses：用于存储每一轮训练时的平均损失值。
acces：用于存储每一轮训练时的平均准确率。
eval_losses：用于存储每一轮测试时的平均损失值。
eval_acces：用于存储每一轮测试时的平均准确率。
SummaryWriter：借助 torch.utils.tensorboard 模块中的 SummaryWriter 类，把训练过程里的损失数据记录到 ./logs 目录下的日志文件中，comment='train-loss' 是对日志的备注信息。
"""

for epoch  in range(num_epoch):
    train_loss=0
    train_acc=0
    model.train()
"""
for epoch in range(num_epoch)：开启一个循环，循环次数由之前定义的 num_epoch 超参数决定，每一次循环代表一轮训练。
train_loss：用于累加每一轮训练中的损失值。
train_acc：用于累加每一轮训练中的正确预测样本数。
model.train()：把模型设置为训练模式，保证像 Dropout、BatchNorm 这类在训练和测试时行为有差异的层，在训练时能正常工作。
"""
#动态修改参数学习率
if epoch%5==0:
    optimizer.param_groups[0]['lr']*=0.9
    print("学习率:{:.6f}".format(optimizer.param_groups[0]['lr']))
"""
每 5 个训练轮次，就把优化器的学习率乘以 0.9 来进行衰减，这样做有助于模型在训练后期更加稳定地收敛。
optimizer.param_groups[0]['lr']：表示优化器中参数组的学习率。
"""
for img, label in train_loader:
    img=img.to(device)
    label=label.to(device)
    """
    for img, label in train_loader：对训练数据加载器进行遍历，每次取出一个批次的图像数据 img 以及对应的标签 label。
    img.to(device) 和 label.to(device)：把数据移到指定的设备（GPU 或者 CPU）上进行计算
    """
    #正向传播
    out=model(img)
    loss=criterion(out,label)
    """
    正向传播：out = model(img) 将图像数据输入到模型中，得到模型的输出；loss = criterion(out, label) 计算模型输出和真实标签之间的损失值。
    """

    #反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    """
    反向传播：optimizer.zero_grad() 把优化器中的梯度信息清空，防止梯度累积；loss.backward() 计算损失函数关于模型参数的梯度；optimizer.step() 根据计算得到的梯度对模型的参数进行更新。
            train_loss += loss.item()：把当前批次的损失值累加到 train_loss 里。
            writer.add_scalar('train', train_loss / len(train_loader), epoch)：把当前轮次的平均训练损失值写入日志文件，方便后续可视化。
    """
    #记录误差
    train_loss+=loss.item()
    #保存loss的数据与epoch数值
    writer.add_scalar('train',train_loss/len(train_loader),epoch)
    #计算分类的准确率
    _,pred=out.max(1)
    num_correct=(pred==label).sum().item()
    acc=num_correct/img.shape[0]
    train_acc+=acc
    """
    计算准确率：out.max(1) 返回每个样本输出中概率最大的类别索引 pred；(pred == label).sum().item() 计算预测正确的样本数；
    """

losses.append(train_loss/len(train_loader))
acces.append(train_acc/len(train_loader))
"""
将当前轮次的平均训练损失值和平均训练准确率分别添加到 losses 和 acces 列表中
"""
#在测试集上检验效果
eval_loss=0
eval_acc=0
#net.eval()#将模式改为预测模式
model.eval()
for img,label in test_loader:#test_loader 是一个数据加载器对象，它会将测试数据集按批次进行划分。这里使用 for 循环遍历 test_loader，每次循环会取出一个批次的图像数据 img 和对应的标签 label。
    img=img.to(device)
    label=label.to(device)#device 是之前定义的设备对象，它可以是 CPU 或者 GPU。to(device) 方法的作用是将张量 img 和 label 移动到指定的设备上进行计算。如果使用 GPU 进行计算，那么数据需要先移动到 GPU 上才能被模型处理。
    img=img.view(img.size(0),-1)
    """
    view() 方法用于调整张量的形状。img.size(0) 表示当前批次中的图像数量，-1 表示让 PyTorch 自动计算该维度的大小。在这里，img 原本是一个四维张量，
    形状通常为 (batch_size, channels, height, width)，
    通过 view() 方法将其转换为二维张量，形状变为 (batch_size, channels * height * width)，这样就可以将图像数据展平为一维向量，以便输入到全连接神经网络中。
    """
    out=model(img)
    """
    将处理后的图像数据 img 输入到已经训练好的模型 model 中，得到模型的预测输出 out。out 是一个二维张量，
    形状为 (batch_size, num_classes)，其中 num_classes 是分类的类别数，对于 MNIST 数据集，num_classes 为 10（数字 0 - 9）。
    """
    loss=criterion(out,label)
    """
    criterion 是之前定义的损失函数，通常在多分类问题中使用交叉熵损失函数 nn.CrossEntropyLoss()。这里将模型的预测输出 out 和真实标签 label 输入到损失函数中，计算当前批次的损失值 loss。"""
    #记录误差
    eval_loss+=loss.item()
    """
    loss.item() 方法用于将损失值 loss（一个零维张量）转换为 Python 标量。将当前批次的损失值累加到 eval_loss 中，以便后续计算整个测试集的平均损失。
    """
    #记录准确率
    _,pred=out.max(1)
    num_correct=(pred==label).sum().item()
    acc=num_correct/img.shape[0]
    eval_acc+=acc
    """
    out.max(1)：返回 out 张量在第 1 个维度（即类别维度）上的最大值和对应的索引。_ 表示忽略最大值，pred 是一个一维张量，包含了每个样本预测的类别索引。
    (pred == label)：这是一个布尔型张量，比较预测的类别索引 pred 和真实标签 label 是否相等，相等的位置为 True，不相等的位置为 False。
    (pred == label).sum().item()：计算布尔型张量中 True 的数量，即预测正确的样本数，并将其转换为 Python 标量。
    acc = num_correct / img.shape[0]：计算当前批次的准确率，即预测正确的样本数除以当前批次的样本数量。
    eval_acc += acc：将当前批次的准确率累加到 eval_acc 中，以便后续计算整个测试集的平均准确率。
    """


eval_losses.append(eval_loss/len(test_loader))
eval_acces.append(eval_acc/len(test_loader))
print("epoch;{},Train Loss:{:.4f},Train Acc:{:.4f},Test Loss:{:.4f},Test Acc:{:.4f}".
      format(epoch,train_loss/len(train_loader),train_acc/len(train_loader),eval_loss/len(test_loader),eval_acc/len(test_loader)))
"""
当前轮次的平均测试损失值和平均测试准确率分别添加到 eval_losses 和 eval_acces 列表中。
使用 print 语句输出当前轮次的训练损失、训练准确率、测试损失和测试准确率。
"""

























