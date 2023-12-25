from model.layer import *
from torch import nn
from torch.nn import functional as F
import torch
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem
import numpy as np
from model import pubchemfp
from math import sqrt
class Self_Attention(nn.Module):
    def __init__(self, input_dim, dim_k, dim_v):
        super(Self_Attention, self).__init__()
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        atten = nn.Softmax(-1)(torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact
        output = torch.bmm(atten, V)
        return atten,output

class FPN(nn.Module):
    def __init__(self,args):
        super(FPN, self).__init__()
        self.fp_2_dim=512
        self.cuda = True
        self.hidden_dim = args.units_dense
        self.args = args
        self.fp_type = 'mixed'
        if self.fp_type == 'mixed':
            self.fp_dim = 1489
        else:
            self.fp_dim = 1024
        if hasattr(args,'fp_changebit'):
            self.fp_changebit = 0
        else:
            self.fp_changebit = None
        self.fc1=nn.Linear(self.fp_dim, self.fp_2_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, smile):
        fp_list=[]
        for i, one in enumerate(smile):
            fp=[]
            mol = Chem.MolFromSmiles(one)
            if self.fp_type == 'mixed':
                fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
                fp_phaErGfp = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)
                fp_pubcfp = pubchemfp.GetPubChemFPs(mol)
                fp.extend(fp_maccs)
                fp.extend(fp_phaErGfp)
                fp.extend(fp_pubcfp)
            else:
                fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fp.extend(fp_morgan)
            fp_list.append(fp)
        if self.fp_changebit is not None and self.fp_changebit != 0:
            fp_list = np.array(fp_list)
            fp_list[:,self.fp_changebit-1] = np.ones(fp_list[:,self.fp_changebit-1].shape)
            fp_list.tolist()
        fp_list = torch.Tensor(fp_list)
        if self.cuda:
            fp_list = fp_list.cuda()
        fpn_out = self.fc1(fp_list)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        return fpn_out
    
def flatten(t):
    t = t.reshape(t.shape[0],-1)
    return t

class Model(nn.Module):
    def __init__(self,hyper):
        super(Model, self).__init__()
        self.num_features = hyper.num_features
        self.units_conv = hyper.units_conv
        self.units_dense = hyper.units_dense
        self.num_layers = hyper.num_layers
        self.pooling = hyper.pooling
        self.outputs = hyper.outputs
        self.dropout = hyper.dropout
        self.alpha = hyper.alpha
        self.nheads = hyper.nheads
        self.model = hyper.model
        self.graphembedding = GraphEmbed()
        self.fpn = FPN(hyper)
        self.tdim = GAT(self.num_features, self.units_conv, self.dropout, self.alpha, self.nheads)
        self.fc_gat1 = nn.Linear(self.units_conv, self.units_dense)
        self.fc_gat2 = nn.Linear(self.units_dense, self.units_dense)
        for num in range(self.num_layers):
            if num == 0:
                setattr(self, "sc_s"+ str(num), GraphSToS(self.units_conv, self.num_features))
                setattr(self, "sc_v" + str(num), GraphVToS(self.units_conv, self.num_features))
                setattr(self, "vc_s" + str(num), GraphSToV(self.units_conv, self.num_features))
                setattr(self, "vc_v" + str(num), GraphVToV(self.units_conv, self.num_features))
                setattr(self, "sc" + str(num), GraphConvS(self.units_conv, self.units_conv, self.units_conv, self.pooling))
                setattr(self, "vc" + str(num), GraphConvV(self.units_conv, self.units_conv, self.units_conv, self.pooling))
            else:
                setattr(self, "sc_s" + str(num), GraphSToS(self.units_conv, self.units_conv))
                setattr(self, "sc_v" + str(num), GraphVToS(self.units_conv, self.units_conv))
                setattr(self, "vc_s" + str(num), GraphSToV(self.units_conv, self.units_conv))
                setattr(self, "vc_v" + str(num), GraphVToV(self.units_conv, self.units_conv))
                setattr(self, "sc" + str(num),
                        GraphConvS(self.units_conv, self.units_conv, self.units_conv, self.pooling, self.dropout, self.alpha, self.nheads))
                setattr(self, "vc" + str(num),
                        GraphConvV(self.units_conv, self.units_conv, self.units_conv, self.pooling, self.dropout, self.alpha, self.nheads))
        self.gat = GAT(self.units_conv, self.units_conv, self.dropout, self.alpha, self.nheads)
        self.graph = GraphGather(pooling=self.pooling)
        self.sc_out1 = nn.Linear(self.units_conv, self.units_dense)
        self.sc_out2 = nn.Linear(self.units_dense, self.units_dense)
        self.vc_out1 = nn.Linear(self.units_conv, self.units_dense)
        self.vc_out2 = nn.Linear(self.units_dense, self.units_dense)
        self.fc1 = nn.Linear(4 * self.units_dense, 2 * self.units_dense)
        self.dropout = nn.Dropout(0.5)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(2 * self.units_dense, self.units_dense)
        self.attention = Self_Attention(self.units_dense, self.units_dense, self.units_dense)
        self.out1 = nn.Linear(3 * self.units_dense, self.units_dense)
        self.out2= nn.Linear(self.units_dense, self.units_dense)
        self.out = nn.Linear(self.units_dense, self.outputs)
    def forward(self, inputs):
        atoms, adjms, dists,batch_s = inputs
        sc, vc = self.graphembedding([atoms,dists])
        sc_fpn = self.fpn(batch_s)
        sc_tdim = self.tdim(sc, adjms)
        sc_tdim = torch.sum(sc_tdim, 1)
        sc_tdim = self.fc_gat1(sc_tdim)
        sc_tdim = F.relu(sc_tdim)
        sc_tdim = self.fc_gat2(sc_tdim)
        sc_tdim = F.relu(sc_tdim)
        for i in range(self.num_layers):
            sc_s = getattr(self, 'sc_s' + str(i))(sc)
            sc_s = F.relu(sc_s)
            sc_v = getattr(self, 'sc_v' + str(i))([vc, dists])
            sc_v = F.relu(sc_v)
            vc_s =  getattr(self, 'vc_s' + str(i))([sc, dists])
            vc_s = F.tanh(vc_s)
            vc_v =  getattr(self, 'vc_v' + str(i))(vc)
            vc_v = F.tanh(vc_v)
            sc = getattr(self, 'sc' + str(i))([sc_s, sc_v, adjms])
            sc = F.relu(sc)
            vc = getattr(self, 'vc' + str(i))([vc_s, vc_v, adjms])
            vc = F.tanh(vc)
        sc = self.gat(sc, adjms)
        sc, vc = self.graph([sc, vc])
        sc_out = self.sc_out1(sc)
        sc_out = F.relu(sc_out)
        sc_out = self.sc_out2(sc_out)
        sc_out = F.relu(sc_out)
        vc_out = self.vc_out1(vc)
        vc_out = F.relu(vc_out)
        vc_out = self.vc_out2(vc_out)
        vc_out = F.relu(vc_out)
        vc_out = flatten(vc_out)
        gcn_out = torch.cat([sc_out, vc_out], -1)
        gcn_out = self.fc1(gcn_out)
        gcn_out = self.dropout(gcn_out)
        gcn_out = self.act_func(gcn_out)
        gcn_out = self.fc2(gcn_out)
        batch = sc_fpn.shape[0]
        sc_fpn = sc_fpn.unsqueeze(1)
        sc_tdim = sc_tdim.unsqueeze(1)
        gcn_out = gcn_out.unsqueeze(1)
        out = torch.cat([sc_fpn, sc_tdim, gcn_out], 1)
        atten, out = self.attention(out)
        out = out.view(batch, -1)
        out = F.relu(out)
        out = self.out1(out)
        out = F.relu(out)
        out = self.out2(out)
        out = F.relu(out)
        out = self.out(out)
        return out
    

def Holo_Mol(hyper):
    mod = Model(hyper)
    return mod
