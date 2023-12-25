from torch import nn
import torch
import torch.nn.functional as F
import copy
import operator
import warnings
warnings.filterwarnings("ignore")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
complate = []
count = 0
class GraphEmbed(nn.Module):
    def __init__(self):
        super(GraphEmbed, self).__init__()
    def forward(self, inputs):
        atoms, distances = inputs
        max_atoms = int(atoms.shape[1])
        atom_feat = int(atoms.shape[-1])
        coor_dims = int(distances.shape[-1])
        vector_features = torch.zeros_like(atoms)
        vector_features = vector_features.unsqueeze(2)
        vector_features = vector_features.repeat(1, 1, coor_dims, 1)
        return [atoms, vector_features]

class GraphSToS(nn.Module):
    def __init__(self,filters,num_features):
        super(GraphSToS, self).__init__()
        self.filters = filters
        self.atom_feat = num_features
        self.W = nn.Parameter(torch.empty(size=(self.atom_feat * 2, self.filters)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.empty(size=(1, self.filters)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, inputs):
        global complate
        global count
        scalar_features = inputs
        max_atoms = int(scalar_features.shape[1])
        atom_feat = int(scalar_features.shape[-1])
        scalar_features = scalar_features.unsqueeze(2)
        scalar_features = scalar_features.repeat(1, 1, max_atoms, 1)
        scalar_features_t = scalar_features.transpose(1,2)
        scalar_features_one = torch.cat([scalar_features, scalar_features_t], -1)
        scalar_features_one = torch.reshape(scalar_features_one, [-1, atom_feat * 2])
        scalar_features = torch.matmul(scalar_features_one, self.W) + self.b
        scalar_features = torch.reshape(scalar_features, [-1, max_atoms, max_atoms, self.filters])
        return scalar_features

class GraphSToV(nn.Module):
    def __init__(self,filters,num_features):
        super(GraphSToV, self).__init__()
        self.atom_feat = num_features
        self.filters = filters
        self.W = nn.Parameter(torch.empty(size=(self.atom_feat * 2, self.filters)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.empty(size=(1, self.filters)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        
    def forward(self, inputs):
        scalar_features, distances = inputs
        max_atoms = int(scalar_features.shape[1])
        atom_feat = int(scalar_features.shape[-1])
        coor_dims = int(distances.shape[-1])
        scalar_features = scalar_features.unsqueeze(2)
        scalar_features = scalar_features.repeat([1, 1, max_atoms, 1])
        scalar_features_t = scalar_features.transpose(1,2)
        scalar_features_one = torch.cat([scalar_features, scalar_features_t], -1)
        scalar_features_one = torch.reshape(scalar_features_one, [-1, atom_feat * 2])
        scalar_features = torch.matmul(scalar_features_one, self.W) + self.b
        scalar_features = torch.reshape(scalar_features, [-1, max_atoms, max_atoms,self.filters])
        scalar_features = scalar_features.unsqueeze(3)
        scalar_features = scalar_features.repeat([1, 1, 1, coor_dims, 1])
        distances = distances.unsqueeze(4)
        distances = distances.repeat([1, 1, 1, 1, self.filters])
        vector_features = torch.mul(scalar_features, distances)
        return vector_features


class GraphVToV(nn.Module):
    def __init__(self,filters,num_features):
        super(GraphVToV, self).__init__()
        self.filters = filters
        self.atom_feat = num_features
        self.W = nn.Parameter(torch.empty(size=(self.atom_feat * 2, self.filters)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.empty(size=(1, self.filters)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        
    def forward(self, inputs):
        vector_features = inputs
        max_atoms = int(vector_features.shape[1])
        atom_feat = int(vector_features.shape[-1])
        coor_dims = int(vector_features.shape[-2])
        vector_features = vector_features.unsqueeze(2)
        vector_features = vector_features.repeat([1, 1, max_atoms, 1, 1])
        vector_features_t = vector_features.transpose(1,2)
        vector_features_one = torch.cat([vector_features, vector_features_t], -1)
        vector_features_one = torch.reshape(vector_features_one, [-1, atom_feat * 2])
        vector_features = torch.matmul(vector_features_one, self.W) + self.b
        vector_features = torch.reshape(vector_features, [-1, max_atoms, max_atoms, coor_dims, self.filters])
        return vector_features

class GraphVToS(nn.Module):
    def __init__(self,filters,num_features):
        super(GraphVToS, self).__init__()
        self.atom_feat = num_features
        self.filters = filters
        self.W = nn.Parameter(torch.empty(size=(self.atom_feat * 2, self.filters)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.empty(size=(1, self.filters)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, inputs):
        vector_features, distances = inputs
        max_atoms = int(vector_features.shape[1])
        atom_feat = int(vector_features.shape[-1])
        coor_dims = int(vector_features.shape[-2])
        vector_features = vector_features.unsqueeze(2)
        vector_features = vector_features.repeat([1, 1, max_atoms, 1, 1])
        vector_features_t = vector_features.transpose(1,2)
        vector_features = torch.cat([vector_features, vector_features_t], -1)
        vector_features = torch.reshape(vector_features, [-1, atom_feat * 2])
        vector_features = torch.matmul(vector_features, self.W) + self.b
        vector_features = torch.reshape(vector_features, [-1, max_atoms, max_atoms, coor_dims, self.filters])
        distances_hat = distances.unsqueeze(4)
        distances_hat = distances_hat.repeat([1, 1, 1, 1, self.filters])
        scalar_features = torch.mul(vector_features,distances_hat)
        scalar_features = torch.sum(scalar_features, -2)
        return scalar_features

class GraphConvS(nn.Module):
    def __init__(self, filters,atom_feat_1,atom_feat_2, pooling='sum', dropout=0.2, alpha=0.2, nheads=8):
        super(GraphConvS, self).__init__()
        self.filters = filters
        self.pooling = pooling
        self.atom_feat_1 = atom_feat_1
        self.atom_feat_2 = atom_feat_2
        self.dropout = dropout
        self.alpha = alpha
        self.nheads = nheads
        self.W = nn.Parameter(torch.empty(size=(atom_feat_1 + atom_feat_2, self.filters)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.empty(size=(1, self.filters)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, inputs):
        scalar_features_1, scalar_features_2, adjacency = inputs
        max_atoms = int(scalar_features_1.shape[1])
        atom_feat_1 = int(scalar_features_1.shape[-1])
        atom_feat_2 = int(scalar_features_2.shape[-1])
        scalar_features = torch.cat([scalar_features_1, scalar_features_2],-1)
        scalar_features = torch.reshape(scalar_features, [-1, atom_feat_1 + atom_feat_2])
        scalar_features = torch.matmul(scalar_features, self.W) + self.b
        scalar_features = torch.reshape(scalar_features, [-1, max_atoms, max_atoms, self.filters])
        adjacency = adjacency.unsqueeze(3)
        adjacency = adjacency.repeat([1, 1, 1, self.filters])
        scalar_features = torch.mul(scalar_features,adjacency)
        if self.pooling == "sum":
            scalar_features = torch.sum(scalar_features, 2)
        elif self.pooling == "max":
            scalar_features = torch.max(scalar_features, 2)
            if len(scalar_features) > 1:
                scalar_features = scalar_features[0]
        elif self.pooling == "avg":
            scalar_features = torch.mean(scalar_features, 2)
        return scalar_features


class GraphConvV(nn.Module):
    def __init__(self, filters,atom_feat_1,atom_feat_2, pooling='sum', dropout=0.2, alpha=0.2, nheads=2):
        super(GraphConvV, self).__init__()
        self.filters = filters
        self.pooling = pooling
        self.atom_feat_1 = atom_feat_1
        self.atom_feat_2 = atom_feat_2
        self.W = nn.Parameter(torch.empty(size=(atom_feat_1 + atom_feat_2, self.filters)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.empty(size=(1, self.filters)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.dropout = dropout
        self.alpha = alpha
        self.nheads = nheads
        self.count = 0

    def forward(self, inputs):
        vector_features_1, vector_features_2, adjacency = inputs
        max_atoms = int(vector_features_1.shape[1])
        atom_feat_1 = int(vector_features_1.shape[-1])
        atom_feat_2 = int(vector_features_2.shape[-1])
        coor_dims = int(vector_features_1.shape[-2])
        vector_features = torch.cat([vector_features_1.cuda(), vector_features_2.cuda()],-1)
        vector_features = torch.reshape(vector_features, [-1, atom_feat_1 + atom_feat_2])
        vector_features = torch.matmul(vector_features, self.W) + self.b
        vector_features = torch.reshape(vector_features, [-1, max_atoms, max_atoms, coor_dims, self.filters])
        adjacency = adjacency.unsqueeze(3)
        adjacency = adjacency.unsqueeze(4)
        adjacency = adjacency.repeat([1, 1, 1, coor_dims, self.filters])
        vector_features = torch.mul(vector_features,adjacency)
        if self.pooling == "sum":
            vector_features = torch.sum(vector_features,2)
        elif self.pooling == "max":
            vector_features = torch.max(vector_features,2)
            if len(vector_features) > 1:
                vector_features = vector_features[0]
        elif self.pooling == "avg":
            vector_features = torch.mean(vector_features, 2)
        return vector_features


class GraphGather(nn.Module):
    def __init__(self,pooling="sum",system="cartesian"):
        super(GraphGather, self).__init__()
        self.pooling = pooling
        self.system = system

    def forward(self, inputs):
        scalar_features, vector_features = inputs
        coor_dims = int(vector_features.shape[2])
        atom_feat = int(vector_features.shape[-1])
        if self.pooling == "sum":
            scalar_features = torch.sum(scalar_features, 1)
            vector_features = torch.sum(vector_features, 1)
        elif self.pooling == "max":
            scalar_features = torch.max(scalar_features,1)
            if len(scalar_features) > 1:
                scalar_features = scalar_features[0]
            vector_features = vector_features.permute([0, 2, 3, 1]).contiguous()
            size = torch.sqrt(torch.sum(vector_features**2, 1))
            idx = torch.argmax(size, -1)
            idx = idx.unsqueeze(1)
            idx = idx.unsqueeze(3)
            idx = idx.repeat([1, coor_dims, 1, 1])
            vector_features = torch.gather(vector_features, -1, idx)
            vector_features = vector_features.squeeze(3)
        elif self.pooling == "avg":
            scalar_features = torch.mean(scalar_features, 1)
            vector_features = torch.mean(vector_features, 1)
        if self.system == "cartesian":
            x, y, z = torch.chunk(vector_features,3, 1)
            r = torch.sqrt(torch.square(x) + torch.square(y) + torch.square(z))
            t = torch.acos(torch.divide(z, r + torch.equal(r, torch.tensor(0).cuda())))
            p = torch.atan(torch.divide(y, x + torch.equal(x, torch.tensor(0).cuda())))
            vector_features = torch.stack([r, t, p], 1)
        return [scalar_features, vector_features]


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False)]
        self.attentions = self.attentions + [GraphAttentionLayer(nhid, nhid, dropout=dropout, alpha=alpha, concat=False) for _ in range(nheads-1)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        for att in self.attentions:
            x = self.leakyrelu(att(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h.double(), self.W.double()).float()
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e.squeeze(-1), zero_vec.squeeze(-1))
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[1] 
        batch = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        return all_combinations_matrix.view(batch, N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
