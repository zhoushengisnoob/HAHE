#encoding:utf-8
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
import scipy.sparse as sp
from collections import defaultdict
from sklearn.metrics import accuracy_score,roc_auc_score,average_precision_score,precision_score,jaccard_similarity_score,recall_score,zero_one_loss


import torch

def make_attribute():
    path1 = '../../dataHIN/DBLP2/PA2.txt'
    path2 = '../../dataHIN/DBLP2/PT2.txt'
    paper_term_map={}
    term_set=set()
    with open(path2) as fp:
        for line in fp.readlines():
            paper=line.strip('\n\r').split()[0]
            term=line.strip('\n\r').split()[1]
            if paper not in paper_term_map:
                paper_term_map[paper]=[]
            paper_term_map[paper].append(term)
            term_set.add(term)
    author_term_map={}
    with open(path1) as fp:
        for line in fp.readlines():
            author=line.strip('\n\r').split()[1]
            paper=line.strip('\n\r').split()[0]
            term_list=paper_term_map[paper]
            if author not in author_term_map:
                author_term_map[author]=[]
            author_term_map[author].append(term_list)

    print('Term number:',len(term_set))
    print('Author number:',len(author_term_map))

    feature_mat=np.zeros((len(author_term_map),len(term_set)))
    for author in author_term_map:
        term_lists=author_term_map[author]
        for term_list in term_lists:
            for term in term_list:
                feature_mat[int(author),int(term)]=1

    res=normalize(feature_mat,norm='l2')
    return res


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)







def multilabel_f1(GT,pred):
    labeled_num_list=torch.sum(GT,dim=1)
    for i in range(len(labeled_num_list)):
        if labeled_num_list[i]==0:
            labeled_num_list[i]=1
    pred_label=torch.zeros_like(GT)
    for i in range(pred.shape[0]):
        pred_label[i,torch.topk(pred[i],int(labeled_num_list[i]))[1]]=1
    #score1 = accuracy_score(y_true=GT,y_pred=pred_label)
    #score1 = roc_auc_score(y_true=GT,y_score=pred)
    score1=0
    score2 = average_precision_score(y_true=GT,y_score=pred,average='micro')
    #score = hamming_loss(GT,pred_label)
    #score = precision_score(GT,pred_label,average='micro')
    #print(GT,pred_label)
    # a_list=[]
    # b_list=[]
    # for i in range(GT.shape[0]):
    #     aa=f1_score(GT[i, :], pred_label[i, :], average='micro')
    #     bb=f1_score(GT[i, :], pred_label[i, :], average='macro')
    #     a_list.append(aa)
    #     b_list.append(bb)
    # return np.mean(a_list),np.mean(b_list)

    #return hamming_score(GT,pred_label),hamming_score(GT,pred_label)
    return f1_score(GT,pred_label,average='micro'),f1_score(GT,pred_label,average='macro'),score2,score1


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def adj2Lap(adj):
    AA=adj+np.ones_like(adj)
    D=np.diag(np.diag(AA))
    DD=np.zeros_like(D)
    [x,y]=np.nonzero(D)
    for i in range(len(x)):
        DD[x[i],y[i]]=np.power(D[x[i],y[i]],-0.5)
    return np.dot(np.dot(DD,AA),DD)


def get_adj(path):
    edge_mat = np.loadtxt(path, dtype=int)
    node_num = int(np.max(edge_mat)) + 1
    adj_mat = np.zeros((node_num, node_num), dtype=int)
    adj_lists = defaultdict(set)
    for i in range(edge_mat.shape[0]):
        adj_mat[edge_mat[i, 0], edge_mat[i, 1]] = 1
        adj_mat[edge_mat[i, 1], edge_mat[i, 0]] = 1
        adj_lists[edge_mat[i, 0]].add(edge_mat[i, 1])
        adj_lists[edge_mat[i, 1]].add(edge_mat[i, 0])
        adj_lists[edge_mat[i, 0]].add(edge_mat[i, 0])
        adj_lists[edge_mat[i, 1]].add(edge_mat[i, 1])
    for i in range(node_num):
        if adj_mat[i,i]==0:
            adj_mat[i, i] = 1
    return adj_mat, adj_lists


def get_adj_mat(path,name):
    adj_mat=loadmat(path)[name].todense()
    for i in range(adj_mat.shape[0]):
        adj_mat[i,i]=1
    adj_lists = defaultdict(set)
    [row,col]=adj_mat.nonzero()
    for i in range(len(row)):
        node1=row[i]
        node2=col[i]
        adj_lists[node1].add(node2)
        adj_lists[node1].add(node1)
        adj_lists[node2].add(node1)
        adj_lists[node2].add(node2)
    return adj_mat,adj_lists


def get_label(path):
    label_mat = np.loadtxt(path, dtype=int)
    node_num = label_mat.shape[0]
    num_class = int(np.max(label_mat[:, 1]) + 1)
    label_list = np.array(label_mat[:, 1])
    return num_class, node_num, label_list

def get_label2(path):
    input_mat = np.loadtxt(path,dtype=int)
    node_num = int(np.max(input_mat[:,0])+1)
    num_class = int(np.max(input_mat[:,1])+1)
    label_mat = np.zeros((node_num,num_class))
    for i in range(input_mat.shape[0]):
        node = input_mat[i, 0]
        label = input_mat[i, 1]
        label_mat[node,label]=1
    return num_class,node_num,label_mat