import torch
import torch.nn as nn
import random
import torch.nn.functional as F


class HomoAttention(nn.Module):
    def __init__(self, features, num_sample,  cuda,  nbr_atten, embed_dim,feature_dim):
        super(HomoAttention, self).__init__()
        self.num_sample = num_sample
        self.cuda = cuda
        self.features = features
        self.nbr_atten = nbr_atten
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.feature2embed = nn.Linear(feature_dim, embed_dim,bias=True)
        self.gcn=True
        if cuda:
            self.feature2embed=self.feature2embed.cuda()
        if self.gcn:
            self.gcn2embed=nn.Linear(embed_dim*2,embed_dim,bias=True)
            if cuda:
                self.gcn2embed=self.gcn2embed.cuda()



    def forward(self, nodes, neighbors):
        sampler = random.sample
        if self.num_sample == None:
            sampled_neighbors = neighbors
        else:
            sampled_neighbors = [
                set(sampler(neighbor, self.num_sample)) if len(neighbor) > self.num_sample else set(neighbor) for
                neighbor in neighbors]


        sampled_neighbors = [(sampled_neighbor | set([nodes[i]])) for i, sampled_neighbor in
                                 enumerate(sampled_neighbors)]

        unique_nodes_list = list(set.union(*sampled_neighbors))
        unique_nodes_dict = {n: i for i, n in enumerate(unique_nodes_list)}
        sampled_neighbor_mat = torch.zeros(len(sampled_neighbors), len(unique_nodes_dict))
        row_indices = [i for i in range(len(sampled_neighbors)) for j in range(len(sampled_neighbors[i]))]
        col_indices = [unique_nodes_dict[n] for neighbor in sampled_neighbors for n in neighbor]
        sampled_neighbor_mat[row_indices, col_indices] = 1

        if self.cuda:
            feature_matrix = self.features(torch.LongTensor(unique_nodes_list)).cuda()
            node_matrix = self.features(torch.LongTensor(nodes)).cuda()
            sampled_neighbor_mat = sampled_neighbor_mat.cuda()
        else:
            feature_matrix = self.features(torch.LongTensor(unique_nodes_list))
            node_matrix = self.features(torch.LongTensor(nodes))

        nbr_embedding = self.feature2embed(feature_matrix)
        node_embedding = self.feature2embed(node_matrix)


        if self.gcn:
            if self.nbr_atten:
                unnormalized_attention = node_embedding.mm(nbr_embedding.t()) * sampled_neighbor_mat
                #unnormalized_attention = F.normalize(unnormalized_attention)
                # selected softmax
                normalized_attention = torch.exp(unnormalized_attention) * sampled_neighbor_mat.div(
                    torch.sum(torch.exp(unnormalized_attention) * sampled_neighbor_mat, dim=1, keepdim=True).repeat(1, len(
                        unique_nodes_list)))
                h_prime = normalized_attention.mm(nbr_embedding)
                h_prime = torch.sigmoid(h_prime)
                h_prime = torch.cat([node_embedding, h_prime], dim=1)
                h_prime = self.gcn2embed(h_prime)
                h_prime = torch.sigmoid(h_prime)
            else:
                pass

        else:
            if self.nbr_atten:
                unnormalized_attention = node_embedding.mm(nbr_embedding.t()) * sampled_neighbor_mat
                #unnormalized_attention = F.normalize(unnormalized_attention)
                normalized_attention = torch.exp(unnormalized_attention) * sampled_neighbor_mat.div(
                    torch.sum(torch.exp(unnormalized_attention) * sampled_neighbor_mat, dim=1, keepdim=True).repeat(1, len(
                        unique_nodes_list)))
                h_prime = normalized_attention.mm(nbr_embedding)
            else:
                pass
        return h_prime

