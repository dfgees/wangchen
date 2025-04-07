
import torch
import torch.nn as nn
import torch.nn.functional as F

from python深度学习第三单元练习 import optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5,1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 36, 3,1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(36*36,128)
        self.fc2=nn.Linear(128,10)
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 36*36)
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return x
net=CNNNet()
net=net.to(device)


#1)定义卷积运算函数
def cust_conv2d(X,K):
    """实现卷积运算"""
    #获取卷积和形状
    h,w=K.shape
    #初始化输入值Y
    Y=torch.zeros(X.shape[0]-h+1,X.shape[1]-w+1)
    #实现卷积运算
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=(X[i:i+h,j:j+w]*K).sum()
    return Y
#2)定义输入及卷积核
X = torch.tensor([[1.0,1.0,1.0,0.0,0.0,0.0],
                  [0.0,1.0,1.0,1.0,0.0,0.0],
                  [0.0,0.0,1.0,1.0,1.0,0.0],
                  [0.0,0.0,1.0,1.0,0.0,0.0],
                  [0.0,1.0,1.0,0.0,0.0,0.0]])
K = torch.tensor([[1.0, 0.0,1.0], [0.0, 1.0,0.0], [1.0, 0.0,1.0]])
result=cust_conv2d(X,K)
print(result)
"""
如何确定卷积核
"""
#1）定义输入与输出
X = torch.tensor([
    [10., 10., 10., 0.0, 0.0, 0.0],
    [10., 10., 10., 0.0, 0.0, 0.0],
    [10., 10., 10., 0.0, 0.0, 0.0],
    [10., 10., 10., 0.0, 0.0, 0.0],
    [10., 10., 10., 0.0, 0.0, 0.0],
    [10., 10., 10., 0.0, 0.0, 0.0]
])
Y = torch.tensor([
    [0.0, 30.0, 30.0, 0.0],
    [0.0, 30.0, 30.0, 0.0],
    [0.0, 30.0, 30.0, 0.0],
    [0.0, 30.0, 30.0, 0.0]
])
#2）训练卷积层
#构造一个二维卷积层，它具有一个输入通道和形状为（3*3）的卷积核
conv2d=nn.Conv2d(1,1,(3,3),bias=False)

#这个二维卷积层使用思维输入和输出格式（批量大小、通风、高度、宽度）
#其中批量大小和通道数都为1
X=X.reshape((1,1,6,6))
Y=Y.reshape((1,1,4,4))
lr=0.001#学习率
#定义损失函数
loss_fn = torch.nn.MSELoss()  # 均方误差损失


for i in range(400):
    Y_pred = conv2d(X)  # 前向传播
    loss = loss_fn(Y_pred, Y)  # 计算损失
    conv2d.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播
    conv2d.weight.data -= lr * conv2d.weight.grad  # 更新卷积核
    if (i + 1) % 100 == 0:
        print(f"epoch:{i + 1} loss:{loss.sum():.4f}")

#3)查看卷积核
conv2d.weight.data.reshape((3,3))

#1)定义多输入通道卷积运算函数
def corr2d_mutil_in(X,K):
    h,w=K.shape[1],K.shape[2]
    value=torch.zeros(X.shape[0]-h+1,X.shape[1]-w+1)
    for x,k in zip(X,K):
        value=value+cust_conv2d(x,k)
    return value
#2)定义输入数据
X = torch.tensor([
    # 通道 0 (5x5)
    [[1., 0., 1., 0., 2.],
     [1., 1., 3., 2., 1.],
     [1., 1., 0., 1., 1.],
     [2., 3., 2., 1., 3.],
     [0., 2., 0., 1., 0.]],

    # 通道 1 (5x5)
    [[1., 0., 0., 1., 0.],
     [2., 0., 1., 2., 0.],
     [3., 1., 1., 3., 0.],
     [0., 3., 0., 3., 2.],
     [1., 0., 3., 2., 1.]],

    # 通道 2 (5x5)
    [[2., 0., 1., 2., 1.],
     [3., 3., 1., 3., 2.],
     [2., 1., 1., 1., 0.],
     [3., 1., 3., 2., 0.],
     [1., 1., 2., 1., 1.]]
])
K = torch.tensor(
    # 输出通道 0 (3个输入通道的卷积核)
    [
        [[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 1.0, 0.0]],  # 输入通道 0 的核
        [[2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]],  # 输入通道 1 的核
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]  # 输入通道 2 的核
    ]
)
#3）计算
result3=corr2d_mutil_in(X,K)
print(result3)

#1）生成输入及卷积核数据
X = torch.tensor([
    # 通道 0 (3x3)
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],

    # 通道 1 (3x3)
    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]],

    # 通道 2 (3x3)
    [[2, 2, 2],
     [2, 2, 2],
     [2, 2, 2]]
])
K = torch.tensor([
    # 输出通道 0 (3个输入通道的1x1核)
    [[[1]],
    [[2]],
    [[3]]],

    # 输出通道 1 (3个输入通道的1x1核)
    [[[4]],
    [[1]],
    [[1]]],

    # 输出通道 2 (3个输入通道的1x1核)
    [[[5]],
    [[3]],
    [[3]]]
])
print(K.shape)
#2)定义卷积函数
def corr2d_mutil_in_out(X,K):
    return torch.stack([corr2d_mutil_in(X,k) for k in K])
result4=corr2d_mutil_in_out(X,K)
print(result4)

#输出大小为5*7
m=nn.AdaptiveMaxPool2d((5,7))
input=torch.randn(1,64,8,9)
output=m(input)
#t输出大小为正方形7*7
m=nn.AdaptiveMaxPool2d(7)
input=torch.randn(1,64,10,9)
output=m(input)
#输出大小为10*7
m=nn.AdaptiveMaxPool2d((None,7))
input=torch.randn(1,64,10,9)
output=m(input)
#输出大小为1*1
m=nn.AdaptiveMaxPool2d((1))
input=torch.randn(1,64,10,9)
output=m(input)
print(output.shape)
"""
构建网络
"""
#1)构建网络
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5,1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, 3,1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(32*32*3,128)
        self.fc2=nn.Linear(128,10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #print(x.shape)
        x=x.view(-1, 36*6*6)
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return x

net=CNNNet()
net=net.to(device)
print(net)
"""
训练模型
"""
#1）选择优化器
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
#2)训练模型
for epoch in range(10):

    running_loss=0.0
    for i,data in enumerate(trainloader,0):
        #获取训练数据
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        #权重参数梯度清零
        optimizer.zero_grad()

        #正向及反向传播
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #显示损失值
        running_loss += loss.item()
        if i % 2000 == 1999:       #print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
"""
LeNet模型的生成代码
"""
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out=F.relu(self.conv1(x))
        out=F.max_pool2d(out, 2)
        out=F.relu(self.conv2(out))
        out=F.max_pool2d(out, 2)
        out=out.view(out.size(0),-1)
        out=F.relu(self.fc1(out))
        out=F.relu(self.fc2(out))
        out=self.fc3(out)
        return out

"""
VGG16模型的实现代码
"""
cfg={
    'VGG16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'VGG19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,'M'],
}
class VGG(nn.Module):
    def __init__(self,vgg_name):
        super(VGG,self).__init__()
        self.features=self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
    def forward(self, x):
        out=self.features(x)
        out = out.view(out.size(0),-1)
        out = self.classifier(out)
        return out
    def _make_layers(self, cfg):
        layers=[]
        in_channels=3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels=x
            layers += [nn.MaxPool2d(kernel_size=1, stride=2)]
            return nn.Sequential(*layers)

















































