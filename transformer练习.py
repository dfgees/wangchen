"""
Transformer架构解析
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable
import copy

#构建Enbedding类来实现文本嵌入层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        #d_model词嵌入维度
        #vocab词表的大小
        super(Embeddings,  self).__init__()
        #定义Embeeding层
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        #x:代表输入进模型的文本通过词汇映射后的数字张量
        return self.lut(x) * math.sqrt(self.d_model)


d_model = 512
vocab = 1000
x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
emb = Embeddings(d_model, vocab)
embr = emb(x)
print(embr)
print(embr.size())

"""
位置编码 器的作用
"""
m = nn.Dropout(p=0.2)
input1 = torch.randn(4, 5)
output = m(input1)
print(output)

x = torch.tensor([1, 2, 3, 4])
y = torch.unsqueeze(x, 0)
print(y)
z = torch.unsqueeze(x, 1)
print(z)


#构建位置编码器的类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        #d_model:代表词嵌入的维度
        #dropout:代表Dropout层的置零比率
        #max_len:每个句子的最大长度
        super(PositionalEncoding, self).__init__()
        #实例化Dropout层
        self.dropout = nn.Dropout(p=dropout)

        #初始化位置编码矩阵，大小是max_len * d_model
        pe = torch.zeros(max_len, d_model)

        #初始化绝对位置矩阵，max_len * 1
        position = torch.arange(0, max_len).unsqueeze(1)

        #定义变换矩阵div_term,跳跃式初始化
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        #将前面定义的变换矩阵进行奇数，偶数的分别赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #将一个二位张量扩充为三位张量
        pe = pe.unsqueeze(0)

        #将位置编码矩阵注册成模型的buffer，这个buffer不是模型中的参数，不跟随优化器同步更新
        #注册成我们的buffer后我们就可以在模型保存后重新加载的时候，将位置编码器和模型参数加载进来
        self.register_buffer('pe', pe)

    def forward(self, x):
        #x:代表文本序列的词嵌入表示
        #首先明确pe的编码太长，将第二个维度，也就是是将max_len对性的那个维度缩小成x的句子长度同等的长度,requires_grad=False表示不参与更新

        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


d_model = 512
dropout = 0.1
max_len = 60

x = embr
pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)
print(pe_result)
print(pe_result.size())

#第一部设置一个画布
plt.figure(figsize=(15, 5))

#实例化PostionalEncoding类对象，词嵌入维度给20，置零比率设置为0
pe = PositionalEncoding(20, dropout=0)

#向pe中传入一个全零初始化的x,相当于展示pe
y = pe(Variable(torch.zeros(1, 100, 20)))

plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())

plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])

# plt.savefig('training_result4.png')
# plt.close()

print(np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9],[10, 11, 12]], k=-1))
print(np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9],[10, 11, 12]], k=0))

#构建掩码张量的函数
def subsequent_mask(size):
    #size:代表掩码张量后两个维度，像成一个方针
    attn_shape = (1, size, size)

    #使用np.ones()先构建一个全1的张量，然后利用np.triu()形成上三角矩阵
    subsequent_mask=np.triu(np.ones(attn_shape), k=1).astype('uint8')

    #使得这个三角矩阵反转
    return torch.from_numpy(1-subsequent_mask)

size=5
"""
这里将 size 设为 5，调用 subsequent_mask 函数后，会生成一个形状为 (1, 5, 5) 的掩码张量。这个张量可以用来对长度为 5 的序列进行掩码操作
"""
sm=subsequent_mask(size)
print(sm)
print(subsequent_mask(20)[0])
plt.figure(figsize=(5, 5))
plt.imshow(subsequent_mask(20)[0])
"""
这里将 size 设为 20，会生成一个形状为 (1, 20, 20) 的掩码张量。通过 [0] 取出第一个维度的元素，得到一个形状为 (20, 20) 的二维数组
"""
plt.savefig('掩码张量图像.png')
plt.close()

"""
mask_filled函数的作用演示
"""
x=Variable(torch.randn(5,5))
print(x)

mask=Variable(torch.zeros(5,5))
print(mask)

y=x.masked_fill(mask==0,1e-9)
print(y)
def attention(query, key, value, mask=None, dropout=None):
    #query,key,value:代表注意力的三个输入量
    #mask掩码张量
    #dropout:传入Dropout实例化对象

    #首先将query最后一个维度提取出来，最后一个维度代表词嵌入的维度
    d_k = query.size(-1)

    #按照注意力计算公式，将query和key的转置进行矩阵乘法，然后除以放缩稀疏
    scores=torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    #判断是否使用掩码张量
    if mask is not None:
        #利用mask_fill方法，将掩码张量和0进行位置的意义比较，如果等于零，替换成一个非常小的数
        scores = scores.masked_fill(mask==0, -1e9)

    #对scoers最后一个维度上进行softmax操作
    p_attn = torch.softmax(scores, dim=-1)

    #判断是否使用dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    #最后一步完成p_attn和value张量的乘法，并返回query注意力表示

    return torch.matmul(p_attn, value), p_attn

query=key=value=pe_result
mask=Variable(torch.zeros(2,4,4))
attn,p_attn=attention(query, key, value,mask=mask)
print(attn)
print(attn.shape)

print(p_attn)
print(p_attn.shape)

"""
tensor.view()演示
"""
x=torch.randn(4,4)
print(x)
print(x.shape)
y=x.view(16)
print(y)
print(y.shape)
z=x.view(-1,8)
print(z)
print(z.shape)

a=torch.randn(1,2,3,4)
print(a)
print(a.shape)

b=a.transpose(1,2)
print(b)
print(b.shape)

c=a.view(1,3,2,4)
print(c)
print(c.shape)

#实现深层拷贝（克隆函数），因为在多头注意机制下，要用到多个结构相同的线性层
#需要使用clone函数将他们一同初始化到一个网络层列表对象中
def clones(module, N):
    #model代表要克隆的目标网络层
    #将model克隆几个
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#实际多头注意力机制的类
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        #head:代表几个头的参数
        #enbedding_dim代表词嵌入的维度
        #dropout：进行Dropout操作时，置零的比率
        super(MultiHeadedAttention, self).__init__()

        #要确认一个事实：多头的数量head需要整除词嵌入的维度embedding_dim
        assert embedding_dim % head == 0
        #得到每个头获得的词向量的维度
        self.d_k = embedding_dim // head

        self.head=head
        self.embedding_dim = embedding_dim

        #获得先行层，要获得四个，分别给Q,K,V，以及最终输出先行层
        self.linears=clones(nn.Linear(embedding_dim, embedding_dim), 4)

        #初始化注意力张量
        self.attn=None

        #初始化dropout对象
        self.dropout=nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        #query,key,value是注意力机制输入的三个张量，mask代表掩码的张量
        #首先判断是否使用掩码张量
        if mask is not None:
            #使用sequeeze将掩码张量进行维度上的扩充，代表多头中的第n个头
            mask=mask.unsqueeze(1)

        #得到batch_size
        batch_size=query.size(0)

        #首先使用zip将网络层和输入数据连接在一起，模型的输出利用view和transpose进行维度和形状上的改变
        query,key,value=\
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1,2)
             for model,x in zip (self.linears, (query, key, value))]

        #将每个头的输出传入到注意力层
        x,self.attn=attention(query,key,value,mask=mask,dropout=self.dropout)

        #得到每个头的计算结果是四位的张量，需要进行形状的转换
        #前面已经将1，2两个维度进行过转置，在这里要重复转置回来
        #注意：经历了transpose()方法后，必须要使用contiguous方法，不然无法使用view()方法
        x=x.transpose(1,2).contiguous().view(batch_size,-1,self.head * self.d_k)

        #最后将x输入到线性层列表中最后一个线性层中进行处理，得到最终的多头注意力结构输出
        return self.linears[-1](x)
#实例化若干参数
head=8
embedding_dim=512
dropout=0.2

#若干输入参数的初始化
query=key=value=pe_result

mask=torch.zeros(2, 4, 4)
mha=MultiHeadedAttention(head,embedding_dim,dropout)
mha_result=mha(query,key,value,mask)
print(mha_result)
print(mha_result.shape)


#构建前馈全连接网络类
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        #d_model:代表词嵌入的维度，同时也是两个线性层的输入维度和输出维度
        #d_ff:代表第一个线性层的输出维度，和第二个线性层的输入维度相同
        #dropout:经过Dropout层处理时，随机置零的比率
        super(PositionwiseFeedForward,self).__init__()

        #定义两层全连接的线性层
        self.w1=nn.Linear(d_model,d_ff)
        self.w2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x):
        #x:代表来自上一层的输出
        #首先将x送入第一个线性层的网络，然后经历一个relu函数的激活，在经过dropout层的处理
        #最后送入第二个线性层
        return self.w2(self.dropout(F.relu(self.w1(x))))

d_model=512
d_ff=64
dropout=0.2

x=mha_result
ff=PositionwiseFeedForward(d_model,d_ff,dropout)
ff_result=ff(x)
print(ff_result)
print(ff_result.shape)

#构建规范化层的类
class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        #features:代表词嵌入的维度
        #eps:一个足够小的正数，用雷规范化计算公式中防止分母为零，
        super(LayerNorm, self).__init__()

        #初始化两个参数张量a2，b2，用于对结果做规范化计算
        #将其用那你。Parameter进行封装，代表他们也是模型中的参数
        self.a2=nn.Parameter(torch.ones(features))
        self.b2=nn.Parameter(torch.zeros(features))

        self.eps=eps

    def forward(self,x):
        #x:代表上一层网络的输出
        #首先对x进行最后一个维度上的求均值操作，同时操持输入维度和输出维度一致
        mean=x.mean(-1,keepdim=True)
        #接着对x进行最后一个维度上求标准差的操作，同时保持输入维度和输出维度保持一致
        std=x.std(-1,keepdim= True)
        #按照规范化公式进行计算并返回
        return self.a2*(x-mean)/(std+self.eps)+self.b2

features=d_model=512
eps=1e-6

x=ff_result
ln=LayerNorm(features,eps)
ln_result=ln(x)
print(ln_result)
print(ln_result.shape)


#构建子层链接结构的类
class SublayerConnection(nn.Module):
    def __init__(self,size,dropout=0.1):
        #size:代表词嵌入的维度
        #dropout代表进行Dropout操作的置零比率
        super(SublayerConnection,self).__init__()
        #实例化一个规范化层的对象
        self.norm=LayerNorm(size)
        #实例化一个dropout的对象
        self.dropout=nn.Dropout(p=dropout)
        self.size=size
    def forward(self,x,sublayer):
        #x:代表上一层传入的张量
        #sublayer:该子层连接中子层函数
        #首先将x进行规范化，然后送入子层函数中处理，处理结果进入dropout层，最后进行残差连接
        return x+self.dropout(sublayer(self.norm(x)))

size=d_model=512
head=8
dropout=0.2

x=pe_result
mask=torch.zeros(2,4,4)
self_attn=MultiHeadedAttention(head,d_model)

sublayer=lambda x:self_attn(x,x,x,mask)
"""
使用 lambda 函数定义子层函数 sublayer，它将输入 x 作为查询、键和值传入多头注意力层，并应用掩码 mask。
"""

sc=SublayerConnection(size,dropout)
sc_result=sc(x,sublayer)
print(sc_result)
print(sc_result.shape)


#构建编码器层的类
class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout=0.1):
        #size:词嵌入的维度
        #self_attn:代表传入的头自注意力子层的实例化对象
        #feed_forward：代表前馈全连接层的实例化对象
        #dropout:进行dropout操作时的置零比率
        super(EncoderLayer,self).__init__()

        #将两个实例化对象和参数传入类中
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.size=size

        #编码器层中有2个子层连接结构，使用clones函数进行操作
        self.sublayer=clones(SublayerConnection(size,dropout),2)

    def forward(self,x,mask):
        #x:代表上一层的传入张量
        #mask:代表掩码张量
        #首先x经过第一个子层连接结构，内部包含多头自注意力机制子层
        #再让我们的张量进行第二个字层的连接结构，其中包括前馈全连接网络
        x=self.sublayer[0](x,lambda x: self.self_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)

size=d_model=512
head=8
d_ff=64
x=pe_result
dropout=0.2

self_attn=MultiHeadedAttention(head,d_model)
ff=PositionwiseFeedForward(d_model,d_ff,dropout)
mask=torch.zeros(2,4,4)

el=EncoderLayer(size,self_attn,ff,dropout)
el_resilt=el(x,mask)
print(el_resilt)
print(el_resilt.shape)

#构建编码器类Encoder
class Encoder(nn.Module):
    def __init__(self,layer,N):
        #layer:代表编码器层
        #N：代表编码器中有几个layer
        super(Encoder,self).__init__()
        #首先使用clones函数克隆N个编码器层放置在self.layers中
        self.layers=clones(layer,N)
        #初始化一个规范化层，作用在编码器的最后面
        self.norm=LayerNorm(layer.size)
    def forward(self,x,mask):
        #x:代表上一层输出的张量
        #mask:代表掩码张量
        #让x依次经过N个编码器层的处理，最后在经过所谓规范化层
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)

zize=d_model=512
d_ff=64
head=8
c=copy.deepcopy
attn=MultiHeadedAttention(head,d_model)
ff=PositionwiseFeedForward(d_model,d_ff,dropout)
dropout=0.2
layer=EncoderLayer(size,c(attn),c(ff),dropout)
N=8
mask=torch.zeros(2,4,4)

en=Encoder(layer,N)
en_result=en(x,mask)
print(en_result)
print(en_result.shape)

#构建解码器层的类
class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout=0.1):
        #size:代表词嵌入的维度
        #self_attn:代表多头自注意力机制的对象
        #src_attn:代表常规的注意力机制对象
        #feed_forward:代表前馈全连接层的对象
        #dropout：代表Dropout的置零比率
        super(DecoderLayer,self).__init__()

        #将参数传入类中
        self.size=size
        self.self_attn=self_attn
        self.src_attn=src_attn
        self.feed_forward=feed_forward
        self.dropout=dropout

        #按照解码器层的结构使用clones函数克隆三个子层连接对象
        self.sublayer=clones(SublayerConnection(size,dropout),3)

    def forward(self,x,memory,source_mask,target_mask):
        #x:代表上一层输入的张量
        #memory:代表编码器的语义存储张量
        #source_mask:原数据的掩码张量
        #target_mask:目标数据的掩码张量
        m=memory

        #第一步让x经历第一个子层，多头自注意力机制的子层
        #采用target_mask，为了将解码是未来的信息遮掩，比如模型解码第二个字符，只能看见第一个字符信息
        x=self.sublayer[0](x,lambda x: self.self_attn(x,x,x,target_mask))

        #第二步，让x经历常规的注意力机制子层，q！=k=v
        #采用source_mask，为了遮掩掉对结果信息无用的数据
        x=self.sublayer[1](x,lambda x: self.src_attn(x,m,m,source_mask))

        #第三步让x经历第三个子层，前馈全连接层
        return self.sublayer[2](x,lambda x: self.feed_forward(x))

#实例化参数
size=d_model=512
head=8
d_ff=64
dropout=0.2

self_attn=src_attn=MultiHeadedAttention(head,d_model,dropout)

ff=PositionwiseFeedForward(d_model,d_ff,dropout)

x=pe_result

memory=en_result

mask=torch.zeros(2,4,4)

source_mask=target_mask=mask
dl=DecoderLayer(size,self_attn,src_attn,ff,dropout)
dl_result=dl(x,memory,source_mask,target_mask)
print(dl_result)
print(dl_result.shape)

#构建解码器类
class Decoder(nn.Module):
    def __init__(self,layer,N):
        #layer:代表解码器层的对象
        #N：代表将layer进行几层的拷贝
        super(Decoder,self).__init__()
        #利用clones函数克隆N个layer
        self.layers=clones(layer,N)
        #实例化一个规范化层
        self.norm=LayerNorm(layer.size)
    #定义逻辑执行过程
    def forward(self,x,memory,source_mask,target_mask):
        #x:代表目标数据的嵌入表示
        #memory：代表编码器的输出张量
        #source_mask：原数据的掩码张量
        #target_mask:目标数据的掩码张量
        #要将x依次经历所有的编码器层处理，最后通过规范化层
        for layer in self.layers:
            x=layer(x,memory,source_mask,target_mask)
        return self.norm(x)

#实例化参数
size=d_model=512
head=8
d_ff=64
dropout=0.2
c=copy.deepcopy
attn=MultiHeadedAttention(head,d_model)
ff=PositionwiseFeedForward(d_model,d_ff,dropout)
layer=DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout)

N=8
x=pe_result
mask=torch.zeros(2,4,4)
source_mask=target_mask=mask

de=Decoder(layer,N)
de_result=de(x,memory,source_mask,target_mask)
print(de_result)
print(de_result.shape)


#构建Generator类


class Generator(nn.Module):
    def __init__(self,d_model,vocal_size):
        #d_model:词嵌入的维度
        #vocal_size：词表的总大小
        super(Generator,self).__init__()
        #定义一个线性层，作用是完成网络输出维度的变换
        self.project=nn.Linear(d_model,vocal_size)

    def forward(self,x):
        #x:代表上一层的输出张量
        #首先将x送入线性层中，让其经历softmax处理
        return F.log_softmax(self.project(x),dim=-1)
#实例化参数
d_model=512
vocal_size=1000
x=de_result

#实例化
gen=Generator(d_model,vocal_size)
gen_result=gen(x)
print(gen_result)
print(gen_result.shape)

#构建编码器——解码器结构类
class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,source_embed,target_embed,generator):
        #encoder:代表编码器对象
        #decoder：代表解码器对象
        #source_embed:代表源函数的嵌入函数
        #target_embed:代表mubiao
        #generator：输出部分类别生成器对象
        super(EncoderDecoder,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=source_embed
        self.tgt_embed=target_embed
        self.generator=generator

    def forward(self,source,target,source_mask,target_mask):
        #source:代表源数据
        #target：代表目标数据
        #scource_mask:代表源数据的掩码张量
        #target_mask:代表目标数据的掩码张量
        return self.decode(self.encode(source,source_mask),source_mask,target,target_mask)

    def encode(self,source,source_mask):
        return self.encoder(self.src_embed(source),source_mask)

    def decode(self,memory,source_mask,target,target_mask):
        #memory:代表经历编码器编码后的输出张量
        return self.decoder(self.tgt_embed(target),memory,source_mask,target_mask)

vocab_size=1000
d_model=512
encoder=en
decoder=de
source_embed=nn.Embedding(vocab_size,d_model)
target_embed=nn.Embedding(vocab_size,d_model)
generator=gen

source=target=torch.LongTensor([[100,2,421,508],[491,998,1,221]])

source_mask=target_mask=torch.zeros(2,4,4)

ed=EncoderDecoder(encoder,decoder,source_embed,target_embed,generator)
ed_result=ed(source,target,source_mask,target_mask)
print(ed_result)
print(ed_result.shape)


def make_model(source_vocab,target_vocab,N=6,d_model=512,d_ff=2048,head=8,dropout=0.2):
    #source_vocab:代表源数据的词汇总数
    #target_vocab:代表目标数据的词汇总数
    #N：代表编码器和解码器堆叠的层数
    #d_model:代表词嵌入的维度
    #d_ff:代表前馈全连接层中变换的维度
    #head：代表注意力机制中的头数
    #dropout：指置零的比率
    c=copy.deepcopy

    #实例化一个多头注意力的类
    attn=MultiHeadedAttention(head,d_model)

    #实例化一个前馈全连接层的网络对象
    ff=PositionwiseFeedForward(d_model,d_ff,dropout)

    #实例化一个位置编码器
    position=PositionalEncoding(d_model,dropout)

    #实例化模型，利用的是EncoderDecoder类
    #编码器的结构里有2个子层，attention层和前馈全连接层
    #解码器的结构中有三个子层，两个attention层和前馈全连接层

    model=EncoderDecoder(
        Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout),N),
        Decoder(DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout),N),
        nn.Sequential(Embeddings(d_model,source_vocab),c(position)),
        nn.Sequential(Embeddings(d_model,target_vocab),c(position)),
        Generator(d_model,target_vocab)
    )
    #初始化整个模型中的参数，判断模型中的参数的维度大于1，，将矩阵初始化成一个服从均匀分布的矩阵
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    return model

source_vocab=11
target_vocab=11
N=6

if __name__=='__main__':
    res=make_model(source_vocab,target_vocab,N)
    print(res)

from pyitcast.transformer_utils import get_std_opt
from pyitcast.transformer_utils import LabelSmoothing
from pyitcast.transformer_utils import SimpleLossCompute

#使用make_model()函数获得模型的实例化对象
model=make_model(V,V,N=2)

#使用工具包get_std_opt获得模型优化器
model_optimizer=get_std_opt(model)

#使用工具包abelSmoothing获得标签平滑对象
criterion=LabelSmoothing(size=V,padding_idx=0,smoothing=0.0)

#使用工具包SimpleLossCompute获得利用标签平滑的结果得到的损失计算方法
loss=SimpleLossCompute(model.generator,criterion,model_optimizer)

crit=LabelSmoothing(size=5,padding_idx=0.5,smoothing=0.5)

predict=torch.FloatTensor([[0,0.2,0.7,0.1,0],
                           [0,0.2,0.7,0.1,0],
                           [0,0.2,0.7,0.1,0],
                           [0,0.2,0.7,0.1,0]])
target=torch.LongTensor([2,1,0])

crit(predict,target)
plt.imshow(crit.true_dist)
plt.savefig('练习.png')
plt.close()


































