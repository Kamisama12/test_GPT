'''
在my_transformer.py的基础上，去掉encoder，只要decoder。
从而实现GPT


-----------------------------------------------------------------------------
version:2023.11.8
-----------------------------------------------------------------------------
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
import tqdm
from tokenizer import tokenizer_ch
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as ddp#用在多机多卡，而且单机多卡比DataParallel更快





n_embed=384#词嵌入的维度，最后单个字符拥有的特征数量
n_head = 6#attention 的个数
block_size = 512#预测输出时候使用的最大上下文长度
dropout=0.0
token_size=None#在获取tokenizer的时候更新
num_layer=6
max_iters = 5000
learning_rate = 3e-4
eval_interval = max_iters//5
batch_size=32
device = 'cuda' if torch.cuda.is_available() else 'cpu'#这个cuda默认会使用cuda:0
# device='cpu'
device_id=[0,1]
eval_iters=200


def get_angle(pos,i_dim,d_model):
    return pos/torch.pow(10000,2*i_dim/d_model)

def pos_embedding(context_size=block_size,n_embed=n_embed):
    #位置编码函数一般保持静态，不进行学习更新
    context_size=min(context_size,block_size)
    with torch.device(device):
        pos_emb=torch.zeros((context_size,n_embed))
        block_tensor=torch.arange(context_size).unsqueeze(1)
        i_dim=torch.arange(0,n_embed,2)
        '''
        #维度是n的时候，我们用i表示偶数和奇数的索引，从0开始到n-1，这时候i只有n的一半减一
        这么处理方便我们利用矩阵乘法一次完成计算和索引
        '''
        if context_size<=1:
            pos_emb[:,0::2]=torch.sin(get_angle(block_tensor,i_dim,n_embed))
        else:
            pos_emb[:,0::2]=torch.sin(get_angle(block_tensor,i_dim,n_embed))
            pos_emb[:,1::2]=torch.cos(get_angle(block_tensor,i_dim,n_embed))
    # print(pos_emb.device)
    return pos_emb

'''
#旋转位置编码
#对Q和K进行位置编码
#Q.size(batch,length,embeded_dimension)   Q.size(batch,length,embeded_dimension)   
'''
def rotary_pos_emb(input:torch.Tensor,seq_len:int):
    with torch.device(device):
        '''
        预先预算编码要用的频率,sin_cos_embed形状->(max_seq_len,d//2))
        '''
        assert 1<input.ndim
        _,_,dim=input.shape
        fres=1.0/torch.pow(1000,torch.arange(0,dim,2)[:(dim//2)]//dim).float()#这里面arange应该不用索引了才对，本来也只有dim//2个元素在里面,size->(d//2,)
        position=torch.arange(seq_len)#(max_seq_len,)
        pos_mul_fres=torch.outer(position,fres).float()#这里计算的是笛卡尔外积，不是向量外积. ->(max_seq_len,d//2)
        sin_cos_embed=torch.polar(torch.ones(pos_mul_fres.size()),pos_mul_fres)#torch.polar(abs,angle)-->out=abs·cos(angle)+abs·sin(angle)j，相等于构建一个复数，用这个方式获取我们频率的sin和cos值
        '''
        重整向量形状适配旋转编码的计算公式的时候让sin_cos_embed可以直接进行broadcast,这份代码里面我的Q K输入形状是(B,T,C)
        '''
        input_=torch.view_as_complex(input.float().reshape(*input.shape[:-1],-1,2))#（B，T，C/2) view_as_complex会消除最后一个维度，这个维度长度是2,变成[a+bj]的复数
        assert sin_cos_embed.shape==(input_.size()[1],input_.size()[2])
        shape=[d if i==1 or i == input_.ndim-1 else 1 for i, d in enumerate(input_.shape)]
        sin_cos_embed.view(*shape)#（1,seq_len,dim//2）
        '''
        按照旋转位置编码的公式进行计算，这里是用复数的形式等效计算，参考https://zhuanlan.zhihu.com/p/647109286
        '''
        out=torch.view_as_real(sin_cos_embed*input_).flatten(-2)#view_as_real之后最后复数部分会变回2维，需要展开对应会矩阵乘法的结果
        return out.type_as(input)


class posembedding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x,context_size):
        return rotary_pos_emb(x,context_size)





class head(nn.Module):
    '''
    Scaled Dot-Product Attention
    '''
    def __init__(self, head_size):
        super().__init__()
        self.query=nn.Linear(n_embed,head_size)
        self.key=nn.Linear(n_embed,head_size)
        self.value=nn.Linear(n_embed,head_size)
        self.pos_emb=posembedding()
        '''#使用不需要进行梯度更新的参数也可以使用下面语句创建缓存区：
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        '''
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        # self.tril=torch.tril(torch.ones((block_size,block_size),device=device))
        # self.tril.requires_grad=False
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)#B,T,C
        q=self.query(x)#B,T,C
        k=self.pos_emb(k,T)
        q=self.pos_emb(q,T)
        '''
        #C是嵌入维度，根据论文内容计算权重，这里得到的形状是(B,T,T)，
        相当于一个二维词表，T表示字符数量，第二个T就表示每个字符和另一个喜爱度。
        如下面所示，展示了模型怎么根据自己的上一个预测出来的字符继续预测下一个字符，
        所以遮罩的时候是对角遮罩的。这一串序列训练的时候是作为y来输入的，是源目标序列右移一位之后的序列。
        序列长度不一，所以一般做了padding到相同长度之后再传进来
        -----------------
           a    b     c   d   <pad>
        a 0.5  -inf -inf -inf
        b 0.5  0.3  -inf -inf
        c 0.5  0.4  0.44 -inf
        d 0.5  0.6  0.8  0.2 
      <pad>
        ------------------

        compute attention scores ("affinities")
        '''
        wei=q@k.transpose(-2,-1)*C**-0.5#wei的形状  B T1 T2
        #这一部分的遮罩应该是给transformer的decoder用的
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))#进行遮罩

        '''
        在padding mask 的帮助下，注意力函数输出的新序列output里头的每个子词都只从序列k 
        （也就是序列q 自己）的前6个实际子词而非<pad> 来获得语义信息。
        当k是encoder的输出而q是decoder的输入时候，是Seq2Seq的注意力机制。
        通常是解码器对编码器的输出序列产生的注意力
        '''
        wei=F.softmax(wei,dim=-1)
        wei=self.dropout(wei)#B T T
        v=self.value(x)#B T C
        output=wei@v 
        return output


class MultiHeadAttention(nn.Module):
    '''
    Multi-Head Attentio
    '''
    def __init__(self, num_heads,head_size) :
        super().__init__()
        self.heads=nn.ModuleList(head(head_size) for _ in range(num_heads))
        self.proj=nn.Linear(n_embed,n_embed)#对应论文Multi-Head Attention里面的linear
        self.dropout=nn.Dropout(dropout)#正则化技术，用于防止神经网络过拟合

    def forward(self,x):
        '''
         #head输出的维度是B T C ，其中C是head_size，即d_dimention//num_head，
         我们将模型维度切割成了多个注意力头进行并行计算
         这里经过多层注意力层之后重新堆叠起来，变回d_dimention维度，即n_embed
        '''
        output=torch.cat([h(x) for h in self.heads],dim=-1)
        output=self.dropout(self.proj(output))
        return output

class FeedFoward(nn.Module):
    '''
    对应论文里面的Feedforward
    '''
    def __init__(self,n_embed):
        super().__init__()
        self.net=nn.Sequential(
            #先升维然后经过ReLu之后再降维
            nn.Linear(n_embed,4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        return self.net(x)

class AddNorm(nn.Module):
    def __init__(self,n_embed) :
        super().__init__()
        '''
        进行层归一化，输入的参数是张量形状或者小于张量形状的维度，且维度上的特征数相同，
        如:对（3，4,5）形状的张量进行layerNorm，参数是5,，（4,5）或者(3,4,5)，区别在于会在相应维度
        计算均值和方差。
        '''
        self.Norm=nn.LayerNorm(n_embed)#层归一化
    def forward(self,x_before,x):#dropout层放在和addNorm连接的层的最后
        output=self.Norm(x_before+x)
        return output

class decodeLayer(nn.Module):
    def __init__(self,n_embed,n_head) :
        super().__init__()
        head_size=n_embed//n_head
        assert head_size*n_head==n_embed,\
        f'请使用能整除的词嵌入维度和注意力头数'

        self.sa1=MultiHeadAttention(n_head,head_size)
        self.addNorm1=AddNorm(n_embed)
        self.sa2=MultiHeadAttention(n_head,head_size)
        self.addNorm2=AddNorm(n_embed)
        self.ffwd=FeedFoward(n_embed)
        self.addNorm3=AddNorm(n_embed)
    def forward(self,x):
        attn1=self.sa1(x)
        out1=self.addNorm1(x,attn1)
        attn2=self.sa2(out1)
        out2=self.addNorm2(out1,attn2)
        out3=self.ffwd(out2)
        output=self.addNorm3(out2,out3)
        return output




class Decoder(nn.Module):
    def __init__(self,n_embed,n_head,num_layer):
        super().__init__()
        self.n_embed=n_embed
        self.tok_emb=nn.Embedding(token_size,n_embed)
        # self.pos_emb=posembedding()
        self.decode=nn.Sequential(*[decodeLayer(n_embed,n_head) for _ in range(num_layer)])
    def forward(self,x):
        B,T=x.shape
        x=self.tok_emb(x)
        x=self.decode(x)
        return x



class Transformer(nn.Module):
    def __init__(self,n_embed,n_head,num_layer):
        super().__init__()
        self.decode=Decoder(n_embed,n_head,num_layer)
        self.ln=nn.Linear(n_embed,token_size)
        # self.softmax=nn.Softmax(dim=1)#softmax留在生成文本的时候再用
    def forward(self,src_ids,trg_ids=None):
        decode_output=self.decode(src_ids)
        logits=self.ln(decode_output)
        if trg_ids is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            trg_ids=trg_ids.view(B*T)
            loss=F.cross_entropy(logits,trg_ids)
        return logits,loss
    '''
    交叉熵cross_entropy的计算，形式一般是：
    我们的预测输出是:x=
    [[1,2,3],
     [4,5,6],
     [7,8,9]],形状是(N,C)，N是样本个数，C是类别总数。
    实际目标输出以序列的形式给出，比如在这里是字符编码，输出序列里面的数字范围不会超过类别总数，因为这是输出索引，个数等于样本数量。
    y=[0,2,1]
    计算过程:
    softmax=(torch.exp(x)/torch.exp(x).sum(dim=1,keepdim=True)
    cross_entropy=-torch.sum(torch.log(softmax[[0,1,2],[2,0,1]]))/x.size(0)
    等于F.cross_entropy(x,y)
    如果使用了onehot矩阵的话可以直接进行矩阵点积来计算
    '''
    def generate(self,src_ids,max_new_token):
        trg_ids=torch.ones((1,1),dtype=torch.long,device=device)
        for _ in range(max_new_token):
            src_idx_cond=src_ids[:,-block_size:]
            logits,loss=self(src_idx_cond)#(B,T,C)
            logits=logits[:,-1,:]#(B,C)#我们关注最后一字的下一个字应该是什么
            prob=F.softmax(logits,dim=1)
            idx_next=torch.multinomial(prob,num_samples=1)#这里不用argmax使用multinomial，增加输出多样性-->(B,1)
            trg_ids=torch.cat((trg_ids,idx_next),dim=1)
        return trg_ids


def get_tokenizer():
    my_tokenizer=tokenizer_ch()
    with open(r'/home/yuzhaohao/LanguageModel/yzh_model/TestTokenizer.json','r') as f:
        token_dic=json.load(f)
        my_tokenizer.encoder=token_dic['encoder']
        my_tokenizer.decoder={int(key):value for key, value in token_dic['decoder'].items()}
    global token_size
    token_size=len(my_tokenizer)
    print(token_size)
    return my_tokenizer
    #将键转化为int数据，dump存储json的时候键全部会强制转换成str

tokenizer=get_tokenizer()    


def padding(src_ids,trg_ids):
    seq_len=max(src_ids,trg_ids,key=len)
    padded_src_ids=src_ids+torch.tensor([0]*(seq_len-len(src_ids)))
    padded_trg_ids=trg_ids+torch.tensor([0]*(seq_len-len(src_ids)))
    #padding之后两个张量的形状会相同
    #一般padding是为了让在一个batch里面的各个样本保持相同的长度，这种一般是因为在作分割的时候没有采取等长分割的方式，他们的样本可能是一组一组的形式。
    #我现在暂时使用的是等长分割的方法，所以在一个batch内每个样本的长度是必定相同的。
    #在做generate的时候可以想象成模型在做的是一个补全文本的任务，我们给出的文本作为src，然后一般用<SOW>特殊字符作为decoder的第一个输入，这时候要考虑两个数据输入的长度不相同
    #一个办法是在head里面做遮罩的时候把遮罩矩阵的两个维度来源分开T1和T2,另一个是进行padding。但是这里又有一个问题，如果将维度分开的话，在训练过程
    #相当于告诉decoder<SOW>的预测是<SOW>，导致推理的时候永远只会生成<SOW>
    #同时如果不做padding的话，推理的时候decoder初始只能看到我们输入的第一个字符，就是<SOW>
    #所以这个函数应该要有两个作用，一个是保证同一个batch内的样本长度相同，另一个保证encoder和decoder的输入长度相同


@torch.no_grad()
def estimate_loss(data):
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(data,0.9,split)
            logits,loss=model(X,Y)#这里如果用了分布式训练封装之后，会返回各个进程的结果,以tensor([10.0056,  9.9915], device='cuda:0')的形式
            losses[k]=loss.mean().item()
        out[split]=losses.mean()
    model.train()
    return out




def read_data(data_path):
    with open(data_path,'r',encoding='utf-8') as f:
        # text=f.read()
        text=json.load(f)
        data=[c['completion']  for c in text]
        data=''.join(data)#连接成长字符串
    data=torch.tensor(tokenizer.encode(data),dtype=torch.long,device=device)
    return data 

def get_batch(data,split_rate,split):
    # data.to(device)
    n=int(split_rate*len(data))
    train_data=data[:n]
    val_data=data[n:]
    data=train_data if split=='train' else val_data
    global block_size
    #留出两个字符位置给起始和终止标志
    max_batch_size=len(data)//(block_size-2)
    assert batch_size<max_batch_size,\
    '训练分词超出最大batch设置，请修改batch_size'
    # test_batch_size=batch_size*2
    ix=torch.randint(len(data)-block_size,(batch_size,))
    '''
    stack和cat区别在于stack会创建一个新维度进行堆叠，而cat沿着原有的维度进行叠加。
    stack堆叠时dim如果设置在还没新建的维度上，最后得到的张量形状会比较奇怪，现在还不太理解，使用的时候尽量只设置在dim=0
    '''
    with torch.device(device):
    #     temp1=lambda i : torch.cat([torch.tensor([1]),data[i:i+block_size-2],torch.tensor([2])])
    #     temp2=lambda i : torch.cat([torch.tensor([1]),data[i+1:i+1+block_size-2],torch.tensor([2])])
        temp1=lambda i : torch.cat([torch.tensor([1]),data[i:i+block_size-1]])
        temp2=lambda i : torch.cat([data[i:i+block_size-1],torch.tensor([2])])
        x=torch.stack([temp1(i) for i in ix])#(B,T)
        y=torch.stack([temp2(i) for i in ix])

    # x,y=x.to(device),y.to(device)
    # print(x.device,y.device)
    return x,y

if __name__=='__main__':
    '''
    做某些验证进行对比时候设定随机种子保证过程可以复现
    '''
    # torch.manual_seed(1337)
    '''
    模型初始化
    '''
    model=Transformer(n_embed,n_head,num_layer)
    # print(model)
    # model.to(device)
    '''
    两个方法进行分布式训练，数据并行，nn.DataParallel直接对model进行封装。
    使用ddp进行封装，但是需要多几行进行初始化设置的代码
    '''
    # model=nn.DataParallel(model,device_ids=device_id)#单机多卡,性能不如ddp
    import os
    dist.init_process_group(backend='nccl')
    local_rank=int(os.environ['LOCAL_RANK'])
    rank=int(os.environ['RANK'])
    num_gpus=torch.cuda.device_count()
    torch.cuda.set_device(rank%num_gpus)#torchrun自动管理rank然后切换设备创建进程。
    device=torch.device('cuda',rank%num_gpus)
    model.to(device)
    model=ddp(model,device_ids=[local_rank],output_device=local_rank)
    print(f'[init]== local rank ={local_rank},global rank : {rank} ==,batch_size : {batch_size} ==')

    '''
    统计模型参数总数
    '''
    num_of_para=sum(p.numel() for p in model.parameters())
    print(f'Total parameters:{num_of_para}')
    
    

    # state_dict={}
    # for param_tensor in model.state_dict():
    #     print(param_tensor,'\t',model.state_dict()[param_tensor].size())
    #     state_dict[param_tensor]=model.state_dict()[param_tensor].size()
    # with open('model_state_dict.json','w') as f:
    #     json.dump([state_dict],f)


        


    
    
    # model.decode.to('cuda:0')
    # print(model)
    # gpu_num=torch.cuda.device_count()
    # print(gpu_num)

    # for name, param in model.named_parameters():
    #     print(f"Parameter {name} on device {param.device}")

    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
    data=read_data('/home/yuzhaohao/LanguageModel/yzh_model/wikipedia-cn-20230720-filtered.json')
    for iter in tqdm.tqdm(range(max_iters),desc='processing'):
        if rank==0 and iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        x,y=get_batch(data,0.9,'train')
        # if int(local_rank) == 0:
        #     x0,y0=x[:batch_size],y[:batch_size]
        #     del x,y
        # elif int(local_rank)==1:
        #     x0,y0=x[batch_size:],y[batch_size:]
        #     del x,y
        #前向和逆向传播
        logits,loss=model(x,y)
        optimizer.zero_grad(set_to_none=True)
        # print(loss）
        loss.mean().backward()
        optimizer.step()
    if rank ==0:
        torch.save(model.module.state_dict(),f'model_module_state_dict.bin')
        # torch.save(model.state_dict(),f'model_state_dict.bin')#单GPU的情况下保存使用这一句
