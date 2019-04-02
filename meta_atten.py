import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CosineSimilarity

cos = CosineSimilarity(dim=1,eps=1e-8)


class Hete_average(nn.Module):
    def __init__(self, meta_atten, cuda,  embed_dim, meta_num, homo_dim, hidden,node_num):
        super(Hete_average, self).__init__()
        self.meta_atten = meta_atten
        self.active = F.tanh
        self.cuda = cuda
        self.embed_dim = embed_dim
        self.meta_num = meta_num
        self.hidden=hidden
        self.homo_dim=homo_dim
        if cuda:
            self.attention=torch.ones(1,meta_num).cuda()
        else:
            self.attention=torch.ones(1,meta_num)
        print(self.attention)

    def forward(self, nodes,homo_embedding_list):
        if self.cuda:
            final_embedding = torch.zeros(len(nodes), self.embed_dim).cuda()
        else:
            final_embedding = torch.zeros(len(nodes),self.embed_dim)

        if self.meta_atten:
            for i in range(self.meta_num):
                final_embedding += self.attention[0,i].repeat(len(nodes), self.homo_dim) * homo_embedding_list[i]
        else:
            pass
        return final_embedding

    
class Hete_self(nn.Module):
    def __init__(self, meta_atten, cuda, embed_dim, meta_num, homo_dim, hidden,node_num):
        super(Hete_self, self).__init__()
        self.meta_atten = meta_atten
        self.meta_num = meta_num
        self.cuda = cuda
        self.embed_dim = embed_dim
        self.homo_dim=homo_dim
        self.homo2hidden = nn.Linear(homo_dim, hidden,bias=True)
        self.hidden = hidden
        if cuda:
            self.homo2hidden=self.homo2hidden.cuda()
            self.out_embedding = nn.Parameter(torch.cuda.FloatTensor(node_num,hidden))
        else:
            self.out_embedding = nn.Parameter(torch.FloatTensor(node_num,hidden))
        nn.init.uniform_(self.out_embedding)

    def forward(self,nodes,homo_embedding_list):
        if self.cuda:
            final_embedding = torch.zeros(len(nodes), self.homo_dim).cuda()
        else:
            final_embedding = torch.zeros(len(nodes),self.homo_dim)
        if self.meta_atten:
            if self.cuda:
                attention=torch.cuda.FloatTensor(len(nodes),self.meta_num)
            else:
                attention=torch.FloatTensor(len(nodes),self.meta_num)

            for i in range(self.meta_num):
                homo_embedding = homo_embedding_list[i]
                hidden_embedding = torch.tanh(self.homo2hidden(homo_embedding))
                attention[:, i] = cos(hidden_embedding,self.out_embedding[nodes])
            attention = F.softmax(attention, dim=1)
            for i in range(self.meta_num):
                final_embedding += attention[:, i].view(-1, 1).repeat(1, self.homo_dim) * homo_embedding_list[i]
            
        else:
            for i in range(self.meta_num):
                homo_embedding = homo_embedding_list[i]
                final_embedding  +=homo_embedding
                final_embedding = final_embedding*1.0/self.meta_num
        return final_embedding


    


# class Hete_self(nn.Module):
#     def __init__(self, meta_atten, cuda, embed_dim, meta_num, homo_dim, hidden,node_num):
#         super(Hete_self, self).__init__()
#         self.meta_atten = meta_atten
#         self.meta_num = meta_num
#         self.cuda = cuda
#         self.embed_dim = embed_dim
#         self.homo_dim=homo_dim
#         self.homo2hidden = nn.Linear(homo_dim, hidden,bias=False)
#         self.hidden = hidden
#         if cuda:
#             self.homo2hidden=self.homo2hidden.cuda()
#             self.self_embed = nn.Parameter(torch.cuda.FloatTensor(1, hidden))
#             self.beta = nn.Parameter(torch.cuda.FloatTensor(1,self.meta_num))
#         else:
#             self.self_embed = nn.Parameter(torch.FloatTensor(1, hidden))
#             self.beta = nn.Parameter(torch.FloatTensor(1,self.meta_num))
#         nn.init.uniform_(self.self_embed)
#         nn.init.uniform_(self.beta)



#     def forward(self, nodes,homo_embedding_list):
#         if self.cuda:
#             final_embedding = torch.zeros(len(nodes), self.homo_dim).cuda()
#         else:
#             final_embedding = torch.zeros(len(nodes),self.homo_dim)
#         if self.meta_atten:
#             if self.cuda:
#                 attention=torch.cuda.FloatTensor(len(nodes),self.meta_num)
#             else:
#                 attention=torch.FloatTensor(len(nodes),self.meta_num)
#             self_embedding = self.self_embed.repeat(len(nodes),1)
#             #self_embedding = self.self_embed[nodes]

#             for i in range(self.meta_num):
#                 homo_embedding = homo_embedding_list[i]
#                 #homo_embedding = F.normalize(homo_embedding)
#                 hidden_embedding = self.homo2hidden(homo_embedding)
#                 #hidden_embedding = F.tanh(hidden_embedding)
#                 #hidden_embedding = F.dropout(hidden_embedding,0.5,training=self.training)
#                 #hidden_embedding = nn.BatchNorm1d(self.hidden).cuda()(hidden_embedding)
#                 #hidden_embedding = F.normalize(hidden_embedding)
#                 #self_embedding = F.normalize(self_embedding)
#                 #attention[:, i] = self.beta[0,i]*torch.diag(self_embedding.mm(hidden_embedding.t()))
#                 attention[:, i] = self.beta[0, i] * cos(hidden_embedding,self_embedding)
#                 #attention[:, i] = cos(self_embedding,hidden_embedding)
#             #print('Beta',self.beta)
#             #print('Before softmax:',torch.mean(attention,dim=0))
#             #attention = F.relu(attention)
#             attention = F.softmax(attention, dim=1)

#             #print('After softmax:',torch.mean(attention,dim=0))
#             for i in range(self.meta_num):
#                 final_embedding += attention[:, i].view(-1, 1).repeat(1, self.homo_dim) * homo_embedding_list[i]
#         else:
#             for i in range(self.meta_num):
#                 homo_embedding = homo_embedding_list[i]
#                 final_embedding  +=homo_embedding
#                 final_embedding = final_embedding*1.0/self.meta_num
#             # final_embedding = torch.zeros(self.meta_num,homo_embedding_list[0].shape[0],homo_embedding_list[0].shape[1]).cuda()
#             # for i in range(self.meta_num):
#             #     final_embedding[i]=homo_embedding_list[i]
#             # final_embedding = torch.max(final_embedding,dim=0)[0]
#         return final_embedding


class Hete_MLP2_atten(nn.Module):
    def __init__(self, meta_atten, cuda,  embed_dim, meta_num, homo_dim, hidden,node_num):
        super(Hete_MLP2_atten, self).__init__()
        self.meta_atten = meta_atten
        self.cuda = cuda
        self.embed_dim = embed_dim
        self.meta_num = meta_num
        #self.beta = nn.Parameter(torch.FloatTensor(1,self.meta_num))
        self.homo_dim=homo_dim
        self.homo2hidden=nn.Linear(homo_dim,hidden,bias=True)
        self.hidden2atten=nn.Linear(hidden,1,bias=True)
        if cuda:
            self.homo2hidden=self.homo2hidden.cuda()
            self.hidden2atten=self.hidden2atten.cuda()
            #self.beta = self.beta.cuda()

    def forward(self, nodes,homo_embedding_list):
        if self.cuda:
            final_embedding = torch.zeros(len(nodes), self.homo_dim).cuda()
        else:
            final_embedding = torch.zeros(len(nodes),self.homo_dim)

        if self.meta_atten:
            attention = torch.FloatTensor(len(nodes), self.meta_num)
            if self.cuda:
                attention = attention.cuda()
            for i in range(self.meta_num):
                homo_embedding = homo_embedding_list[i]
                #homo_embedding = F.normalize(homo_embedding)
                attention[:, i] =self.hidden2atten(F.tanh(self.homo2hidden(homo_embedding))).squeeze(dim=1)
            attention = F.softmax(attention, dim=1)
            #print(torch.mean(attention,dim=0,keepdim=True))
            for i in range(self.meta_num):
                final_embedding += attention[:, i].view(-1, 1).repeat(1, self.homo_dim) * homo_embedding_list[i]
        else:
            for i in range(self.meta_num):
                final_embedding +=homo_embedding_list[i] * 1.0 / self.meta_num
        return final_embedding


# class Hete_MLP1_atten(nn.Module):
#     '''
#     single layer MLP
#     '''
#     def __init__(self, feature_list,num_sample,nbr_atten,meta_atten, cuda, feature_dim, embed_dim, meta_num, homo_dim, hidden,adj_lists_list):
#         super(Hete_MLP1_atten, self).__init__()
#         self.meta_atten = meta_atten
#         self.cuda = cuda
#         self.feature_dim = feature_dim
#         self.embed_dim = embed_dim
#         self.meta_num = meta_num
#         self.adj_lists_list=adj_lists_list
#         self.encoder = nn.ModuleList()
#         self.homo_dim=homo_dim
#         for i in range(meta_num):
#             self.encoder.append(HomoAttention(features=feature_list[i], num_sample=num_sample,  cuda=cuda,
#                                               nbr_atten=nbr_atten, embed_dim=homo_dim, feature_dim=feature_dim))
#         self.homo2atten=nn.Linear(homo_dim,1,bias=True)
#         if cuda:
#             self.homo2atten=self.homo2atten.cuda()
#
#     def forward(self, nodes):
#         if self.cuda:
#             final_embedding = torch.zeros(len(nodes), self.homo_dim).cuda()
#         else:
#             final_embedding = torch.zeros(len(nodes),self.homo_dim)
#
#         if self.meta_atten:
#             attention = torch.zeros(len(nodes), self.meta_num)
#             if self.cuda:
#                 attention = attention.cuda()
#                 homo_embedding_tensor=torch.cuda.FloatTensor(self.meta_num,len(nodes),self.homo_dim)
#             else:
#                 homo_embedding_tensor = torch.FloatTensor(self.meta_num, len(nodes), self.homo_dim)
#             for i in range(self.meta_num):
#                 homo_embedding = self.encoder[i].forward(nodes,[list(self.adj_lists_list[i][int(node)]) for node in nodes])
#                 homo_embedding=F.normalize(homo_embedding)
#                 homo_embedding_tensor[i]=homo_embedding
#                 attention[:, i] =F.tanh(self.homo2atten(homo_embedding)).squeeze(dim=1)
#             attention = F.softmax(attention, dim=1)
#             print(torch.mean(attention,dim=0,keepdim=True))
#             for i in range(self.meta_num):
#                 homo_embedding=homo_embedding_tensor[i]
#                 final_embedding += attention[:, i].view(-1, 1).repeat(1, homo_embedding.shape[1]) * homo_embedding
#         else:
#             pass
#         return final_embedding
#
# class Hete_MLP12_atten(nn.Module):
#     def __init__(self, feature_list,num_sample,nbr_atten,meta_atten, cuda, feature_dim, embed_dim, meta_num, homo_dim, hidden,adj_lists_list):
#         super(Hete_MLP12_atten, self).__init__()
#         self.meta_atten = meta_atten
#         self.cuda = cuda
#         self.embed_dim = embed_dim
#         self.meta_num = meta_num
#         self.adj_lists_list=adj_lists_list
#         self.encoder = nn.ModuleList()
#         self.W_list = nn.ParameterList()
#         self.homo_dim=homo_dim
#         for i in range(meta_num):
#             self.encoder.append(HomoAttention(features=feature_list[i], num_sample=num_sample, cuda=cuda,
#                                               nbr_atten=nbr_atten, embed_dim=homo_dim, feature_dim=feature_dim))
#         self.homo2embed=nn.Linear(homo_dim,embed_dim,bias=True)
#         self.embed2hidden=nn.Linear(embed_dim,hidden,bias=True)
#         self.hidden2atten=nn.Linear(hidden,1,bias=False)
#         if cuda:
#             self.homo2embed = self.homo2embed.cuda()
#             self.embed2hidden = self.embed2hidden.cuda()
#             self.hidden2atten = self.hidden2atten.cuda()
#
#
#     def forward(self, nodes):
#         if self.cuda:
#             final_embedding = torch.zeros(len(nodes), self.embed_dim).cuda()
#         else:
#             final_embedding = torch.zeros(len(nodes),self.embed_dim)
#
#         if self.meta_atten:
#             attention = torch.FloatTensor(len(nodes), self.meta_num)
#             if self.cuda:
#                 attention = attention.cuda()
#             homo_embedding_tensor=torch.cuda.FloatTensor(self.meta_num,len(nodes),self.embed_dim)
#             for i in range(self.meta_num):
#                 homo_embedding = self.encoder[i].forward(nodes,[list(self.adj_lists_list[i][int(node)]) for node in nodes])
#                 homo_embedding = F.normalize(homo_embedding,p=2,dim=1)
#                 homo_embedding = self.homo2embed(homo_embedding)
#                 homo_embedding_tensor[i]=homo_embedding
#                 attention[:, i] = self.hidden2atten(F.tanh(self.embed2hidden(homo_embedding))).squeeze(dim=1)
#             attention = F.softmax(attention, dim=1)
#             print(torch.mean(attention,dim=0,keepdim=True))
#             for i in range(self.meta_num):
#                 homo_embedding=homo_embedding_tensor[i]
#                 final_embedding += attention[:, i].view(-1, 1).repeat(1, homo_embedding.shape[1]) * homo_embedding
#         else:
#             pass
#         return final_embedding
#
# class Hete_concate(nn.Module):
#     def __init__(self, feature_list,num_sample,nbr_atten,meta_atten, cuda, feature_dim, embed_dim, meta_num, homo_dim, hidden,adj_lists_list):
#         super(Hete_concate, self).__init__()
#         self.meta_atten = meta_atten
#         self.cuda = cuda
#         self.feature_dim = feature_dim
#         self.embed_dim = embed_dim
#         self.meta_num = meta_num
#         self.adj_lists_list=adj_lists_list
#         self.encoder = nn.ModuleList()
#         self.homo_dim=homo_dim
#         for i in range(meta_num):
#             self.encoder.append(HomoAttention(features=feature_list[i], num_sample=num_sample,  cuda=cuda,
#                                               nbr_atten=nbr_atten, embed_dim=homo_dim, feature_dim=feature_dim))
#
#         self.concate2hidden=nn.Linear(meta_num*homo_dim,hidden,bias=True)
#         self.hidden2embed = nn.Linear(hidden,embed_dim,bias=True)
#         if cuda:
#             self.concate2hidden=self.concate2hidden.cuda()
#             self.hidden2embed=self.hidden2embed.cuda()
#
#
#     def forward(self, nodes):
#         if self.cuda:
#             final_embedding = torch.zeros(len(nodes), self.homo_dim).cuda()
#         else:
#             final_embedding = torch.zeros(len(nodes),self.homo_dim)
#
#         if self.meta_atten:
#             cat_embedding = torch.zeros(len(nodes),self.homo_dim)
#             for i in range(self.meta_num):
#                 homo_embedding = self.encoder[i].forward(nodes,[list(self.adj_lists_list[i][int(node)]) for node in nodes])
#                 homo_embedding=F.normalize(homo_embedding)
#                 if i==0:
#                     cat_embedding =homo_embedding
#                 else:
#                     cat_embedding=torch.cat((cat_embedding,homo_embedding),dim=1)
#             final_embedding = self.concate2hidden(cat_embedding)
#             #final_embedding = F.tanh(final_embedding)
#             final_embedding = self.hidden2embed(final_embedding)
#             #final_embedding = F.tanh(final_embedding)
#
#         else:
#             pass
#         return final_embedding
#
# class Hete_MLP3_atten(nn.Module):
#     '''
#     MLP to transform and then learn embedding
#     '''
#     def __init__(self, feature_list,num_sample,nbr_atten,meta_atten, cuda, feature_dim, embed_dim, meta_num, homo_dim, hidden,adj_lists_list):
#         super(Hete_MLP3_atten, self).__init__()
#         self.meta_atten = meta_atten
#         self.active = F.tanh
#         self.cuda = cuda
#         self.feature_dim = feature_dim
#         self.embed_dim = embed_dim
#         self.meta_num = meta_num
#         self.adj_lists_list=adj_lists_list
#         self.encoder = nn.ModuleList()
#         self.W_list = nn.ModuleList()
#         self.homo_dim=homo_dim
#         for i in range(meta_num):
#             self.encoder.append(HomoAttention(features=feature_list[i], num_sample=num_sample,  cuda=cuda,
#                                               nbr_atten=nbr_atten, embed_dim=homo_dim, feature_dim=feature_dim))
#             self.W_list.append(nn.Linear(self.homo_dim,1,bias=True))
#
#         if cuda:
#             self.W_list=self.W_list.cuda()
#
#     def forward(self, nodes):
#         if self.cuda:
#             final_embedding = torch.zeros(len(nodes), self.homo_dim).cuda()
#         else:
#             final_embedding = torch.zeros(len(nodes),self.homo_dim)
#
#         if self.meta_atten:
#             attention = torch.zeros(len(nodes), self.meta_num)
#             if self.cuda:
#                 attention = attention.cuda()
#             homo_embedding_tensor=torch.cuda.FloatTensor(self.meta_num,len(nodes),self.homo_dim)
#             for i in range(self.meta_num):
#                 homo_embedding = self.encoder[i].forward(nodes,[list(self.adj_lists_list[i][int(node)]) for node in nodes])
#                 homo_embedding = F.normalize(homo_embedding,p=2,dim=1)
#                 homo_embedding_tensor[i]=homo_embedding
#                 attention[:, i] = F.tanh(self.W_list[i](homo_embedding)).squeeze(dim=1)
#             attention = F.softmax(attention, dim=1)
#             print(torch.mean(attention,dim=0,keepdim=True))
#             for i in range(self.meta_num):
#                 homo_embedding=homo_embedding_tensor[i]
#                 final_embedding += attention[:, i].view(-1, 1).repeat(1, homo_embedding.shape[1]) * homo_embedding
#         else:
#             pass
#         return final_embedding
