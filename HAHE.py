import torch
import torch.nn as nn
import argparse
import numpy as np
import random
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import f1_score,average_precision_score,hamming_loss
from tensorboardX import SummaryWriter
from utils import make_attribute,multilabel_f1,normalize_adj,adj2Lap,get_adj,get_adj_mat,get_label,get_label2
from scipy.io import loadmat

from nbr_atten import HomoAttention
from meta_atten import Hete_self


class HAHE_train(nn.Module):
    def __init__(self,  homo_encoder_list,hete_encoder,cuda, num_class, embed_dim,meta_num,adj_lists_list):
        super(HAHE_train, self).__init__()
        self.cuda = cuda
        self.homo_encoder_list=homo_encoder_list
        self.hete_encoder = hete_encoder
        self.meta_num = meta_num
        self.adj_lists_list=adj_lists_list
        if cuda:
            self.embed2class = nn.Linear(embed_dim,num_class,bias=True).cuda()
        else:
            self.embed2class = nn.Linear(embed_dim, num_class, bias=True)


    def forward(self, nodes):
        homo_embedding_list=[]
        for i in range(self.meta_num):
            homo_embedding_list.append(self.homo_encoder_list[i](nodes,[list(self.adj_lists_list[i][int(node)]) for node in nodes]))
        embedding = self.hete_encoder.forward(nodes,homo_embedding_list)
        return embedding

    def loss(self, nodes, labels):
        scores = self.embed2class(self.forward(nodes))
        if len(labels.shape) == 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
            #criterion = nn.MultiLabelSoftMarginLoss()



        if self.cuda:
            labels = labels.cuda()
        loss = criterion(scores, labels.squeeze())
        return loss


class HOMO_train(nn.Module):
    def __init__(self, features, feature_dim, adj, cuda, embed_dim, num_sample,  nbr_atten, num_class):
        super(HOMO_train, self).__init__()
        self.cuda = cuda
        self.adj = adj
        self.encoder = HomoAttention(features=features, num_sample=num_sample,  cuda=self.cuda,
                                     feature_dim=feature_dim, nbr_atten=nbr_atten, embed_dim=embed_dim)

        if cuda:
            self.embed2class = nn.Linear(embed_dim,num_class,bias=True).cuda()
        else:
            self.embed2class = nn.Linear(embed_dim, num_class, bias=True)

    def forward(self, nodes):
        embedding = self.encoder.forward(nodes, [list(self.adj[int(node)]) for node in nodes])
        return embedding

    def loss(self, nodes, labels):
        embedding = self.forward(nodes)
        scores = self.embed2class(embedding)
        if len(labels.shape)==1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
        if self.cuda:
            labels = labels.cuda()

        loss = criterion(scores, labels.squeeze())
        return loss





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=666, help='Random seed.')
    parser.add_argument('--num_sample', type=int, default=100, help='Number of neighbors to sample')
    parser.add_argument('--homo_dim', type=int, default=128, help='Dimension of Homo embedding')
    parser.add_argument('--embed_dim', type=int, default=128, help='Dimension of embedding')
    parser.add_argument('--batch', type=int, default=512, help='Batch Size')
    parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units')
    parser.add_argument('--epoch', type=int, default=200, help='Number of Epoches')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning Rate')
    parser.add_argument('--decay', type=float, default=0.001, help='Weight decay for Adam')
    parser.add_argument('--meta', type=str, default=None, help='Single Path for training')
    parser.add_argument('--data', type=str, default='DBLP', help='Data set to use')
    parser.add_argument('--cuda', action='store_false', default=True, help='Using GPU for training')
    parser.add_argument('--nbr_atten', action='store_false', default=True, help='Learning attention for neighbors')
    parser.add_argument('--meta_atten', action='store_false', default=True, help='Learning attention for meta path')
    parser.add_argument('--board', action='store_true', default=False, help='Add log for tensorboard')
    parser.add_argument('--train', type=float, default=0.3, help='training percent')
    args = parser.parse_args()
    print('Learning Neighbor Attention:', args.nbr_atten)
    print('Learning Meta Attention:', args.meta_atten)
    print('Using GPU for training:', args.cuda)
    print('Tensorboard available:', args.board)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    train_percent = args.train
    print('Train percent:', train_percent)



    if args.data=='DBLP':
        label_dir = '../../dataHIN/DBLP/label2.txt'
        num_class, num_nodes, labels = get_label(label_dir)
        print('Nodes:', num_nodes,'Class:',num_class)
        #meta_list = ['APVPA']
        meta_list=['APA','APPA','APVPA']
        meta_num = len(meta_list)

    elif args.data=='douban':
        label_dir = '../../dataHIN/douban/label2.txt'
        num_class,num_nodes,labels = get_label(label_dir)
        print('Nodes:', num_nodes,'Class:', num_class)
        meta_list = ['mum2','mdm2']
        meta_num = len(meta_list)

    elif args.data == 'patent':
        label_dir = '../../dataHIN/patent/label2.txt'
        num_class, num_nodes, labels = get_label(label_dir)
        print('Nodes:', num_nodes, 'Class:', num_class)
        meta_list = ['pap', 'pp', 'pip']
        #meta_list=['pap']
        meta_num = len(meta_list)

    elif args.data == 'Douban_full':
        label_dir = '../../dataHIN/Douban_full/label2.txt'
        num_class, num_nodes, labels = get_label2(label_dir)
        print('Nodes:', num_nodes, 'Class:', num_class)
        meta_list = ['mam', 'mdm', 'mum']
        meta_num = len(meta_list)

    elif args.data =='yelp':
        label_dir = '../../dataHIN/yelp/label2.txt'
        num_class, num_nodes, labels = get_label(label_dir)
        print('Nodes:', num_nodes, 'Class:', num_class)
        #meta_list = ['BRB', 'BRURB','BRKRB']
        meta_list = ['BRURB','BRKRB']
        meta_num = len(meta_list)

    elif args.data=='imdb':
        label_dir = '../../dataHIN/imdb/label.txt'
        num_class, num_nodes, labels = get_label2(label_dir)
        print('Nodes:', num_nodes, 'Class:', num_class)
        meta_list = ['mam','mum']
        meta_num = len(meta_list)

    if args.meta == None:
        adj_lists_list = []
        feature_list=[]
        homo_encoder_list=[]
        print('Training multiple meta path')
        for meta_path in meta_list:
            if args.data=='DBLP':
                dir = '../../dataHIN/DBLP/' + meta_path + '_unweighted.txt'
            elif args.data=='douban':
                dir = '../../dataHIN/Douban/' + meta_path + '.txt'
            elif args.data == 'patent':
                dir = '../../dataHIN/patent/' + meta_path + '.txt'
            elif args.data=='Douban_full':
                dir = '../../dataHIN/Douban_full/' + meta_path + '.mat'
            elif args.data == 'yelp':
                dir = '../../dataHIN/yelp/' + meta_path + '.mat'
            elif args.data == 'imdb':
                dir = '../../dataHIN/imdb/' + meta_path + '.txt'

            if args.data=='Douban_full' or args.data =='yelp':
                adj, adj_lists = get_adj_mat(dir,meta_path)
                adj = F.normalize(torch.FloatTensor(adj),p=2,dim=1)
            else:
                adj, adj_lists = get_adj(dir)
                adj = F.normalize(torch.FloatTensor(adj),p=2,dim=1)


            adj_lists_list.append(adj_lists)
            fea = nn.Embedding(num_nodes, num_nodes)
            fea.weight = nn.Parameter(torch.FloatTensor(adj), requires_grad=False)
            feature_list.append(fea)
            num_features=num_nodes
            print('Nodes:',num_nodes,'Features:',num_features)

        for i in range(meta_num):
            homo_encoder = HomoAttention(features=feature_list[i], num_sample=args.num_sample,  cuda=args.cuda,  nbr_atten=args.nbr_atten, embed_dim=args.homo_dim,feature_dim=num_nodes)
            homo_encoder.load_state_dict(torch.load('../../dataHIN/'+args.data+'/'+meta_list[i]+'_para_'+str(round(train_percent*10))+'.pkl'))
            homo_encoder_list.append(homo_encoder)
        hete_encoder = Hete_self(cuda=args.cuda,meta_atten=args.meta_atten,embed_dim=args.embed_dim,hidden=args.hidden,homo_dim=args.homo_dim,meta_num=meta_num,node_num=num_nodes)
        hahe_train = HAHE_train(homo_encoder_list=homo_encoder_list,hete_encoder=hete_encoder,cuda=args.cuda, num_class=num_class, embed_dim=args.embed_dim,meta_num= meta_num,adj_lists_list=adj_lists_list)

    else:
        print('Training for single meta path')
        print('Meta path:',args.meta)
        if args.data == 'DBLP':
            dir = '../../dataHIN/DBLP/' + args.meta + '_unweighted.txt'
        elif args.data == 'douban' :
            dir = '../../dataHIN/douban/' + args.meta + '.txt'
        elif args.data == 'Douban_full' :
            dir = '../../dataHIN/Douban_full/' + args.meta + '.mat'
        elif args.data == 'patent':
            dir = '../../dataHIN/patent/' + args.meta + '.txt'
        elif args.data == 'yelp':
            dir = '../../dataHIN/yelp/' + args.meta + '.mat'
        elif args.data == 'imdb':
            dir = '../../dataHIN/imdb/' + args.meta + '.txt'


        if args.data == 'Douban_full' or args.data =='yelp':
            adj,adj_lists = get_adj_mat(dir,args.meta)
            adj = F.normalize(torch.FloatTensor(adj),p=2,dim=1)
        else:
            adj, adj_lists = get_adj(dir)
            adj = F.normalize(torch.FloatTensor(adj),p=2,dim=1)

        num_features=num_nodes
        features = nn.Embedding(num_nodes,num_features)
        features.weight = nn.Parameter(torch.FloatTensor(adj),requires_grad=False)

        hahe_train = HOMO_train(features=features, feature_dim=num_features, adj=adj_lists, num_class=num_class,
                                cuda=args.cuda, embed_dim=args.embed_dim,
                                num_sample=args.num_sample, nbr_atten=args.nbr_atten)


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, hahe_train.parameters()), lr=args.lr,weight_decay=args.decay)

    rand_indices = np.random.permutation(num_nodes)
    train_index = list(rand_indices[:round(num_nodes * train_percent)])
    test_index = list(rand_indices[round(num_nodes * train_percent):])

    for epoch in range(args.epoch):
        hahe_train.train()
        optimizer.zero_grad()
        batch_nodes = train_index[:args.batch]
        np.random.shuffle(train_index)
        if args.data=='DBLP' or args.data=='patent' or args.data == 'yelp' or args.data=='douban' :
            loss = hahe_train.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        elif args.data=='Douban_full' or args.data == 'imdb':
            loss = hahe_train.loss(batch_nodes, Variable(torch.FloatTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()

        hahe_train.eval()
        test_output = hahe_train.forward(test_index)
        test_output = hahe_train.embed2class(test_output)
        if args.data == 'DBLP' or args.data =='patent' or args.data =='yelp' or args.data == 'douban':
            f1_score_micro = f1_score(labels[np.array(test_index)], test_output.data.cpu().numpy().argmax(axis=1),
                                    average='micro')
            f1_score_macro = f1_score(labels[np.array(test_index)], test_output.data.cpu().numpy().argmax(axis=1),
                                    average='macro')
            print( epoch,loss.detach().cpu().numpy(), f1_score_micro,f1_score_macro)
        elif args.data == 'Douban_full' or args.data=='imdb':
            [f1_score_micro,f1_score_macro,score1,score2] =multilabel_f1(torch.FloatTensor(labels[np.array(test_index)]),test_output.data.cpu())
            print(epoch,loss.detach().cpu().numpy(), f1_score_micro,f1_score_macro,score1,score2)


    if args.meta!=None:
        all_embedding = hahe_train.forward(list(range(num_nodes)))
        np.save('../../dataHIN/'+args.data+'/'+args.meta+'_'+str(round(train_percent*10)), all_embedding.detach().cpu().numpy())
        torch.save(hahe_train.encoder.state_dict(),'../../dataHIN/'+args.data+'/'+args.meta+'_para_'+str(round(train_percent*10))+'.pkl')


    out_embedding=True
    if out_embedding:
        all_embedding = hahe_train.forward(list(range(num_nodes)))
        np.savetxt('HAHE_'+args.data+'_res.npy',all_embedding.detach().cpu().numpy())
    
#     if out_embedding:
#         all_embedding = hahe_train.forward(list(range(num_nodes)))
#         with open('dblp_embedding_res.tsv','w') as wp:
#             for i in range(all_embedding.shape[0]):
#                 for j in range(128):
#                     wp.write(str(all_embedding.detach().cpu().numpy()[i,j])+'\t')
#                 wp.write('\n')

