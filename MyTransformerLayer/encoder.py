import unittest
import math

import torch
from torch import nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, dim: int = 32, head_dim_kq:  int = 32, head_dim_v: int = 32, nHeads: int = 8, mlp_dim = 120):
        super(Transformer, self).__init__()
        self.dim = dim
        self.head_dim_kq = head_dim_kq
        self.head_dim_v = head_dim_v
        self.nHeads = nHeads
        self.head_dim = 2*head_dim_kq + head_dim_v # osszesen ennyi dimenzio tartozik egy input vektorhoz egy head-nel
        # arra kell, hogy minden head egyesevel legyen feldolgozva

        self.inner_dim = nHeads * (2*head_dim_kq + head_dim_v) # hány dimenzió reprezentálja a szekvencia egy elemét
        # minden egyes head egy key, value és query vektorokbol áll

        self.to_qkv = nn.Linear(dim, self.inner_dim)
        self.dropout1 = nn.Dropout()
        self.back_to_dim = nn.Linear(self.nHeads * head_dim_v, dim)
        self.dropout2 = nn.Dropout()

        self.linear_to_hidden = nn.Linear(dim, mlp_dim)
        self.dropout2 = nn.Dropout()

        self.linear_from_hidden_to_dim = nn.Linear(mlp_dim, dim)
        self.dropout2 = nn.Dropout()

    def forward(self, x): # x -> [nB, seqLength, dim]
        nB, seqLength, dim = x.shape
        frozen = x # rezidualis atkotes
        x = self.to_qkv(x)
        # attention map kiszamitasa

        stacked_vs = None

        for head in range(self.nHeads): # minden head-re a a value ertekek beszorzasa a megfelelo attention-map ekkel
            if stacked_vs is None:
                stacked_vs = self.processOneHead(x, head) # qkv
            else:
                stacked_vs = torch.concat((stacked_vs, self.processOneHead(x, head)), 2)

        stacked_vs = self.dropout1(stacked_vs)
        stacked_vs = self.back_to_dim(stacked_vs)

        stacked_vs += frozen

        frozen = stacked_vs  # rezidualis atkotes
        stacked_vs = self.dropout2(stacked_vs)
        stacked_vs = self.linear_to_hidden(stacked_vs)
        stacked_vs = self.linear_from_hidden_to_dim(stacked_vs)
        stacked_vs += frozen
        # x = x + frozen
        return stacked_vs

    def processOneHead(self, x, head):
        # szettordeljuk a qkv-t
        keyoffset = head*self.head_dim
        queryoffset = head*self.head_dim + self.head_dim_kq
        valueoffset = head*self.head_dim + 2*self.head_dim_kq
        qs = x[:, :, keyoffset:keyoffset + self.head_dim_kq]
        ks = x[:, :, queryoffset:queryoffset + self.head_dim_kq]
        vs = x[:, :, valueoffset:valueoffset + self.head_dim_v]
        qs = torch.transpose(qs, 1, 2) # transzponaljuk a query vektorokat, hogy attention map-et szamoljunk
        attn = torch.matmul(ks, qs)
        attn = F.softmax(attn, dim = 2) # utolso dimenzio menten lesz a softmax
        new_vs = torch.matmul(attn, vs)
        return new_vs

def main():
    x = torch.randn(10, 3, 32)
    print(x.shape)
    t = Transformer()
    x = t(x)
    print(x.shape)

if __name__ == "__main__":
    main()
