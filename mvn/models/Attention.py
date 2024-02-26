import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np


class SEnet(nn.Module):
    def __init__(self,channels,ratio=16):
        super(SEnet, self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        # 经过两次全连接层，一次较小，一次还原
        # self.fc1 = nn.Sequential(
        #     nn.Linear(channels,channels//ratio,False),
        #     nn.ReLU())
        # self.fc2 =nn.Sequential(
        #     nn.Linear(channels//ratio, channels, False),
        #     nn.Sigmoid())
        self.fc=nn.Sequential(
            nn.Linear(channels,channels//ratio,False),
            nn.ReLU(),
            nn.Linear(channels//ratio, channels, False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_=x.size() #取出batch size和通道数
        # b,c,w,h->b,c,1,1->b,c 以便进行全连接
        avg=self.avgpool(x).view(b,c)
        #b,c->b,c->b,c,1,1 以便进行线性加权
        fc=self.fc(avg).view(b,c,1,1) 
        # fc1=self.fc1(avg)
        # fc2=self.fc2(fc1).view(b,c,1,1) 
        
        return fc*x

adj1 = torch.tensor([[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,1],
                     [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
                     [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0],
                     [0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0],
                     [0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
                     [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1]
                    ])
# a = torch.tensor([[[2],[1]]])
# b = torch.tensor([[[1,2],[3,4]]])
# print(a.shape)
# print(b.shape)
# c = (np.dot(b,a)).reshape(1,2,1)
# print(c)
# print(c.shape)
# print(a*b)

class CGCA_branch(nn.Module):
    # adj是邻接矩阵
    def __init__(self, channels, channels_attention, adj=adj1):
        super(CGCA_branch, self).__init__()
        self.num_joints = 17
        self.adj = adj
        self.g = (self.adj > 0)  # 权重矩阵G = 17 * 17,邻边设置为true
        self.e = nn.Parameter(torch.zeros(1, len(self.g.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)  # e矩阵全部填充为1

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 分组卷积块
        self.Group = nn.Sequential(
            nn.Conv2d(channels, channels_attention, kernel_size=1, bias=False),
            nn.Conv2d(channels_attention, channels_attention, kernel_size=1, bias=False,
                      groups=channels_attention // self.num_joints)
        )

        self.fc1 = nn.Linear(channels_attention, self.num_joints, False)
        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.num_joints, channels, False),
            nn.Sigmoid()
        )



    def forward(self, x):
        # 生成一个与adj形状相同的全1张量,然后与-9e15相乘
        adj = -9e15 * torch.ones_like(self.adj).to(x.device)
        # 邻接矩阵中与节点邻接的点 = eij, 那么不与节点相邻的点就=-9e15
        adj[self.g] = self.e  # 元素运算 ⊙
        adj = F.softmax(adj, dim=1)  # 按行使用softmax进行归一化

        n, c, _, _ = x.size()  # 取出batch size，视角数和通道数

        x1 = self.Group(x) # nx255xhxw
        n, c1, _, _ = x1.size()

        avg = self.avgpool(x1).view(n, c1)
        fc1 = self.fc1(avg).reshape(n, self.num_joints,1)

        adj = adj.reshape(1, self.num_joints, self.num_joints)
        # print(adj.shape)
        # print(fc1.shape)

        print((torch.matmul(adj,fc1)).shape)
        gc = torch.matmul(adj,fc1).to(x.device).view(n,self.num_joints)
        print("gc:",gc.shape)
        #

        # (n,c)
        fc2 = self.fc2(gc).view(n, c, 1, 1)

        return fc2 


