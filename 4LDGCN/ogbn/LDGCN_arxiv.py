# -*- coding: utf-8 -*-
"""
LDGCN on ogbn-arxiv
Layer-diverse Negative Sampling for Graph Neural Networks (TMLR 2024)
"""

import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)


import torch
import numpy as np 
import networkx as nx
import math
import pickle

from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import argparse

parser = argparse.ArgumentParser(description='OGBN-Arxiv (LDGCN)')
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--sample_rate', type=float, default=0.001, help='Negative sampling nodes rate')
parser.add_argument('--save', action='store_true', help='save training logs under Result/')
args = parser.parse_args()
print(args)



dataset = PygNodePropPredDataset(name='ogbn-arxiv',transform=T.ToSparseTensor())

data = dataset[0]  # Get the first graph object.

print()
print(f'Dataset: {dataset}:')  
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
# print(f'Number of training nodes: {data.train_mask.sum()}')
# print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
print(f'Contains self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

data.adj_t = data.adj_t.to_symmetric()
print(data.adj_t)
row, col, edge_attr = data.adj_t.t().coo()
edge_index = torch.stack([row, col], dim=0)
print(edge_index)
print(edge_index.shape)
data.edge_index = edge_index
data.y = data.y.squeeze()
print(data)
print(f'Is undirected: {data.is_undirected()}')

split_idx = dataset.get_idx_split()
train_idx = split_idx['train']
val_idx = split_idx['valid']
test_idx = split_idx['test']

train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
test_mask[test_idx] = True
val_mask[val_idx] = True
print(train_mask.sum())
print(test_mask.sum())
print(val_mask.sum())
data.train_mask = train_mask
data.test_mask = test_mask
data.val_mask = val_mask
num_nodes = data.num_nodes
print(data) 
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.4f}')

"""# Get the largest connected subgraph """

from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
print('===========================================================================================================')

def GetMaxConnectGraph(OrginData):
  G = to_networkx(OrginData, to_undirected=True)
  Subnode = max(nx.connected_components(G), key=len)
  print(f'Number of nodes of max-connect Graph: {len(Subnode)}')

  SubGraph = G.subgraph(Subnode)
  #Change from networkx to pytorch
  PyTSubGraph = from_networkx(SubGraph)
  SubGraph = to_networkx(PyTSubGraph, to_undirected=True)
  # # diameter = nx.algorithms.distance_measures.diameter(SubGraph)
  # diameter = 19
  # print(f'Diameter of max-connect Graph: {diameter}')

  #Make new sub-graph data
  SubnodeList = list(Subnode)
  PyTSubGraph.test_mask = data.test_mask[SubnodeList]
  PyTSubGraph.train_mask = data.train_mask[SubnodeList]
  PyTSubGraph.val_mask = data.val_mask[SubnodeList]
  PyTSubGraph.x = data.x[SubnodeList]
  PyTSubGraph.y = data.y[SubnodeList]
  return PyTSubGraph,SubGraph

if data.has_isolated_nodes():
  PyTSubGraph, SubGraph = GetMaxConnectGraph(data)
  print(PyTSubGraph)
  num_nodes = PyTSubGraph.num_nodes
  edge_index = PyTSubGraph.edge_index
else:
  PyTSubGraph = data
  SubGraph = to_networkx(data, to_undirected=True)
  num_nodes = data.num_nodes
  edge_index = data.edge_index

# print(adjdense.shape)
avgDegree = edge_index.shape[1]/num_nodes
print(f'PyTSubGraph: {PyTSubGraph}')
print(f'Average degree: {avgDegree}')
print('===========================================================================================================')

"""# Get Candidate Sets"""

'''
  Using Shortest Path Dictionary 
  Save the last node of a path:
      1) length >2 
      2) the degree of the beginning node of this path >10
'''
def GetNodeLenDict(SPathDict,DegreeList,AvgGgree):
  LenDict = defaultdict(list)
  for node in SPathDict:
    templen = SPathDict[node]
    if templen > 2 and DegreeList[node]>AvgGgree: # Exclusion of first-order nearest neighbors
      LenDict[templen].append(node)
  return LenDict

import time
import random
from networkx import single_source_shortest_path,single_source_shortest_path_length
import collections
from collections import defaultdict

begin = time.time()
SPEndNodeDict = defaultdict(dict)
DegreeList = []

'''
  1 calculate Degree of every node in sparse Adj matrix
'''
out_list = edge_index[0].tolist()
d = collections.Counter(out_list)
DegreeList = []
for i in range(num_nodes):
  DegreeList.append(d[i])
end = time.time()
print(f'Degree List time: {end - begin}')

'''
  2 Random Select 50% nodes to get shortest path of every node
    Save the last node of path in SPEndNodeDict

    Example: 
    1) Random select central Node "6" as "i" in "SelCenNodeList"
    2) SPathDict: Get all shortest path of "6" to other Nodes
    3) LenDict: Get last nodes diction for every lenth
        LenDict.keys() : dict_keys([3, 4, 5, 6, 7, 8, 9])
        LenDict[4] : [16384, 16387, 16399, 18, 16405....]
'''
SelCenNodeList = random.sample(range(0, num_nodes), int(num_nodes*args.sample_rate))
os.makedirs("cache", exist_ok=True)
pkl_path = os.path.join("cache", "R"+str(args.sample_rate*100)+"%OGBN-Arxiv_SP_EndNodeDict.pkl")
if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as handle:
            SPEndNodeDict = pickle.load(handle)
else:   
        SPEndNodeDict = defaultdict(dict)
        t1 = time.time()
        for i in SelCenNodeList:
            SPathDict = single_source_shortest_path_length(SubGraph, i,cutoff=7)
            LenDict = GetNodeLenDict(SPathDict,DegreeList,avgDegree)
            SPEndNodeDict[i]=LenDict
        t2 = time.time()
        print(f'Calculate SPEndNodeDict time: {t2-t1}')
        with open(pkl_path, "wb") as file:
            pickle.dump(SPEndNodeDict, file)


# load pickle module


# create a binary pickle file 
# f = open("MAX0.1%ogbn-arxiv_SP_EndNodeDict.pkl","wb")

# # # write the python object (dict) to pickle file
# pickle.dump(SPEndNodeDict,f)

# close file
# f.close()
# print("Save SP_EndNodeDict Done")
# # import pickle
# with open('MAX0.1%ogbn-arxiv_SP_EndNodeDict.pkl', 'rb') as handle:
#     SPEndNodeDict = pickle.load(handle)



SelCenNodeList = list(SPEndNodeDict.keys())
print(f'Select node num: {len(SelCenNodeList)}')

"""# Shortest Path for node candidates Section"""

def GetCandidateNode(SPEndNodeDict,SelCenNodeList,SelectNum):
  candidates = defaultdict(list)
  c = 0
  for node in SelCenNodeList:
    SelNode = []
    select_num = SelectNum
    for i in SPEndNodeDict[node]:
      if len(SPEndNodeDict[node][i])>0:
        randomSel = random.sample(SPEndNodeDict[node][i], 1)
        # print(randomSel)
        SelNode.extend(randomSel)
        select_num = select_num -1
      if select_num ==0:
        break
    candidates[node].extend(SelNode) 
  return candidates

begin = time.time()
candidates = GetCandidateNode(SPEndNodeDict,SelCenNodeList,5)
end = time.time()
print(f'Get candidates time: {end-begin}')
# print(candidates)

AvgNodeDgree = int(data.num_edges / data.num_nodes)+1
def GetCandiSet(SelCenNodeList,candidates):
  Adjcandidates = defaultdict(list)
  for i in SelCenNodeList:
    AdjList = []
    for CenterIndex in candidates[i]:
        EdgeIndex = torch.nonzero(edge_index[0]==CenterIndex)# get adj node index of central node
        EdgeIndex = torch.reshape(EdgeIndex, (-1,))
        AjdIndex = edge_index[1][EdgeIndex]
        AjdIndex = AjdIndex.tolist()
        if len(AjdIndex)>AvgNodeDgree:
            AjdIndex =  random.sample(AjdIndex,AvgNodeDgree)
        AdjList.extend(AjdIndex)
        AdjList.append(CenterIndex)
    Adjcandidates[i] = AdjList
  return Adjcandidates

begin = time.time()
Adjcandidate = GetCandiSet(SelCenNodeList,candidates)
end = time.time()
print(f'Get Adjcandidate time: {end-begin}')
# print(Adjcandidate)
print('===========================================================================================================')

"""# Community Section"""

# from networkx.algorithms.community.label_propagation import label_propagation_communities
# LabelComu = label_propagation_communities(SubGraph)

from networkx.algorithms import community 
NumbCom = 100
Iteration =100
FluidCom = community.asyn_fluidc(SubGraph,NumbCom,Iteration)
LabelComu = list(FluidCom)
print(len(LabelComu))

def GetCommunityMatrix(label_community,num_nodes):
  lcomuMatrix = torch.zeros(len(label_community), num_nodes)
  com = 0 
  for c in label_community:
    for ele in c:
      lcomuMatrix[com][ele] = 1
    com = com+1
  return lcomuMatrix

# Get community matrix / Every node belong to different community
LComuMatrix = GetCommunityMatrix(LabelComu,num_nodes) 
print(LComuMatrix.shape)
print(LComuMatrix.shape[1])
# print(LComuMatrix.sum(1)[0:10])

def elementary_symmetric_poly(k,N,eigenvalues,device):
  e_M = 1e-6*torch.ones([N+1,k+1]).to(device) #e_M = e^n_l
  e_M[:,0] = 1
  # print(e_M.device)
  # print(eigenvalues.device)
  for l in range(1,k+1):
    v = e_M[:-1,l-1] * eigenvalues
    e_l = torch.cumsum(v, dim=0)
    e_M[1:,l] = e_l
  return e_M

def k_DPP_Sampling_VectorIndex(k,N,eigenvalues,device):
  e_M = elementary_symmetric_poly(k,N,eigenvalues,device) # e_M[N+1,k+1]
  J = []
  l = k
  u = torch.rand([N,k]).to(device)
  tt = torch.unsqueeze(eigenvalues, 1)
  a =  tt*e_M[:-1,:-1]/e_M[1:,1:]
  for n in range(N,0,-1):
    if l == 0 : 
      # print("n",n)
      break
    if u[n-1,l-1] < a[n-1,l-1]:
      J.append(n-1)
      l = l-1
  return J

def torch_proj_dpp_sampler_eig_KuTa12(eig_vecs, size=None, IsMask=False, Mask = None):
    """ Sample from :math:`\\operatorname{DPP}(K)` using the 
          eigendecomposition of the similarity kernel :math:`K`.
    It is based on the orthogonalization of the selected eigenvectors.
    :param eig_vals:
        Collection of eigen values of the similarity kernel :math:`K`.
    :type eig_vals:
        list
    :return: list
        A sample from :math:`\\operatorname{DPP}(K)`.    
    """

    # rng = check_random_state(random_state)

    # Initialization
    V = eig_vecs.clone()

    N, rank = V.shape  # ground set size / rank(K)
    if size is None or size > rank:  # full projection DPP
        size = rank
    # else: k-DPP with k = size

    sampl = torch.zeros(size).int()  # sample list

    # Phase 1: Already performed!
    # Select eigvecs with Bernoulli variables with parameter the eigvals

    # Phase 2: Chain rule
    norms_2 = (V*V).sum(axis=1) # ||V_i:||^2
    # print(rank)
    # Following [Algo 1, KuTa12], the aim is to compute the orhto 
    #  complement of the subspace spanned by the selected eigenvectors 
    #  to the canonical vectors \{e_i ; i \in Y\}. We proceed recursively.
    for it in range(size):
        # j = np.random.choice(N, p=np.abs(norms_2) / (rank - it))
        probs = torch.abs(norms_2) /(rank - it)
        # print(probs)

        if torch.isnan(probs).any():
          break
        # if IsMask:
        #   maskprob = Mask * probs
        #   probs = maskprob/maskprob.sum()

        dist=torch.distributions.categorical.Categorical(probs=probs)
        j = dist.sample()
        sampl[it] = j
        if it == size - 1:
            break

        # Cancel the contribution of e_i to the remaining vectors that is, 
        #  find the subspace of V that is orthogonal to \{e_i ; i \in Y\}
        # Take the index of a vector that has a non null contribution on e_j
        k = torch.where(V[j, :] != 0)[0][0]
        # Cancel the contribution of the remaining vectors on e_j, but stay 
        #  in the subspace spanned by V i.e. get the subspace of V orthogonal 
        #  to \{e_i ; i \in Y\}
        V -= torch.outer(V[:, k] / V[j, k], V[j, :])

        # V_:j is set to 0 so we delete it and we can derive an orthononormal 
        #  basis of the subspace under consideration
        # V, _ = la.qr(np.delete(V, k, axis=1), mode='economic')
        tempIndex = list(range(V.shape[1]))
        tempIndex.remove(k)
        V_delet_k = V[:,tempIndex]

        V, _ =  torch.linalg.qr(V_delet_k)
        norms_2 = (V*V).sum(axis=1)   # ||V_i:||^2

    return sampl.tolist()

"""# GCN Section"""

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import gather_csr, scatter
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]
        # edge_index has shape [2, E]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)  # [N, ]
        deg_inv_sqrt = deg.pow(-0.5)   # [N, ]
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j
            
    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

import gc 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DiffGroupNorm
import random


import time

class LDGCN(torch.nn.Module):
    def __init__(self,seed, device, hidden_channels,num_layers,num_nodes,LabelComu,LComuMatrix,DegreeList):
        super(LDGCN, self).__init__()
        torch.manual_seed(seed)
        self.device = device
        self.num_layers = num_layers
        self.CONVs = torch.nn.ModuleList()
        self.DPPCONVs = torch.nn.ModuleList()
        self.NegRates = nn.ParameterList()
        self.Norms = torch.nn.ModuleList()
        for layer in range(self.num_layers-1):
            if layer == 0:
                self.CONVs.append(GCNConv(dataset.num_features, hidden_channels))
                self.DPPCONVs.append(GCNConv(dataset.num_features, hidden_channels))
                self.NegRates.append(nn.Parameter(torch.FloatTensor([1])))
            else:
                self.CONVs.append(GCNConv(hidden_channels, hidden_channels))
                self.DPPCONVs.append(GCNConv(hidden_channels, hidden_channels))
                self.NegRates.append(nn.Parameter(torch.FloatTensor([1])))
            self.Norms.append(torch.nn.BatchNorm1d(hidden_channels))
        self.CONVs.append(GCNConv(hidden_channels, dataset.num_classes))
        self.DPPCONVs.append(GCNConv(hidden_channels, dataset.num_classes))
        self.NegRates.append(nn.Parameter(torch.FloatTensor([1])))
        # self.Norms.append(DiffGroupNorm(dataset.num_classes,50))

        self.LabelComu = LabelComu
        self.LComuMatrix = LComuMatrix
        self.DegreeList = DegreeList
        # self.adjdense = adjdense
        self.num_nodes = num_nodes
        # According to experimental experience, setting NegRate to 1~3 can achieve better results
        self.NegRate = 0.3
        self.DPPedge_index_list = []
        self.ComuFeatureM = 0
        self.ClusterList = []
        self.LayerDppIndex = []


    def forward(self, x, edge_index,Adjcandidate,DPPedge_index_list,train_neg_rate):
        self.DPPedge_index_list = DPPedge_index_list
        for layer in range(self.num_layers):
          self.ComuFeatureM = self.GetComuFeatureMatrix(x,).to(self.device)
          x_temp = x.clone()
          posi_x = self.CONVs[layer](x, edge_index)
          if len(self.DPPedge_index_list) < self.num_layers:
            with torch.no_grad():
              Dppedge_index = self.GetDppMatrix(x_temp,Adjcandidate)
              self.DPPedge_index_list.append(Dppedge_index)
          else:
            Dppedge_index = self.DPPedge_index_list[layer]

          nega_x = self.DPPCONVs[layer](x_temp, Dppedge_index.to(self.device))
          if train_neg_rate: #NegRate is a trainable parameter
            x = posi_x - self.NegRates[layer] * nega_x
          else:
            x = posi_x - self.NegRate * nega_x

          # x = self.Norms[layer](x)
          if layer < (self.num_layers-1):
            x =self.Norms[layer](x)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            # if self.training:
            #   cl, _ = KMeans(x, self.LComuMatrix.shape[0], verbose=False)
            #   self.ClusterList.append(cl)
        return x,self.DPPedge_index_list,self.ClusterList
        
    def GetComuFeatureMatrix(self,x):
        temp = 0
        n = 0
        ComuFeatureM = torch.zeros(len(self.LabelComu), x.shape[1]) #[number of Comu, number of features]
        for c in self.LabelComu:
          for ele in c:
            temp= temp + x[ele]
          ComuFeatureM[n] = temp / len(c)
          n = n+1
        return ComuFeatureM

    def GetDppMatrix(self,x,Adjcandidate):
        # for sparse adj matrix
        beginList = []
        endList = []
        valueList = []
        LstDppIndex = defaultdict(list)
        for node in Adjcandidate:
          nodeDegree = self.DegreeList[node]
          if nodeDegree > len(Adjcandidate[node]):
              end = Adjcandidate[node]
          else:
              #Get community index of central node "node"
              cumIndex = torch.nonzero(self.LComuMatrix[:,node]).reshape(-1)
              #Get feature of community of "node"
              CNodeFeature = self.ComuFeatureM[cumIndex.item()]
              #index list of cummunity of every node
              AjdIndex = Adjcandidate[node]
              cumiList = torch.nonzero(self.LComuMatrix[:,AjdIndex])[:,0]

              Index = self.Rek_dpp_kernel(node, AjdIndex,x, cumiList, CNodeFeature, nodeDegree)
              LstDppIndex[node].extend(Index)
              AjdIndex = np.array(AjdIndex)
              end = AjdIndex[Index]
              if type(end) == np.int64:
                end = [end]
          value = [1]*len(end)
          begin = [node]*len(end)
          beginList.extend(begin)
          endList.extend(end)
          valueList.extend(value)
        i = torch.LongTensor([beginList,endList])   #row, col
        v = torch.FloatTensor(valueList)    #data
        sparseMatrix = torch.sparse.FloatTensor(i, v, torch.Size([num_nodes,num_nodes]))  
        self.LayerDppIndex.append(LstDppIndex)
        return sparseMatrix._indices()

    def Rek_dpp_kernel(self, GivenNode, AjdIndex, x, cumiList,CNodeFeature,k_number):
        # print(x)
        NodeWeight = x[AjdIndex]
        ComuWeight = self.ComuFeatureM[cumiList]
        NodecosSim = self.CosSimilarity(NodeWeight,NodeWeight)
        ComucosSim = self.CosSimilarity(NodeWeight,ComuWeight)
        
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        NodeFeatureCos = cos(CNodeFeature,(NodeWeight.sum(0)/NodeWeight.shape[0]))

        CNodeFeMatrix = torch.ones_like(ComuWeight)
        CNodeFeMatrix = CNodeFeMatrix * CNodeFeature
        QijMatrix =  NodeFeatureCos * self.CosSimilarity(CNodeFeMatrix,ComuWeight) 

        Lmatrix = QijMatrix*torch.mm(ComucosSim, ComucosSim.transpose(0,1)) * torch.exp(NodecosSim-1)* QijMatrix.transpose(0,1)

        Lmatrix = Lmatrix + 0.01*torch.eye(Lmatrix.shape[0]).to(self.device)
        EigVal, EigVec = torch.linalg.eigh(Lmatrix)
        ind_selected = k_DPP_Sampling_VectorIndex(k_number,Lmatrix.shape[0],EigVal.real,self.device)
        
    
        if len(self.DPPedge_index_list)>=1:
          EigVec = self.GetMaskEigVector(GivenNode, x, AjdIndex,EigVec)
          DppEigVec = EigVec[:, ind_selected].real
          samples = torch_proj_dpp_sampler_eig_KuTa12(DppEigVec,k_number,self.device) #,IsMask = True, Mask = probMask)
        else:
          DppEigVec = EigVec[:, ind_selected].real
          samples = torch_proj_dpp_sampler_eig_KuTa12(DppEigVec,k_number)
      
        return samples

    def GetMaskEigVector(self, GivenNode, x, AjdIndex,EigVec):
        LstNegNodesList = []
        # Get Node List of last layer negative sample of GivenNode
        tempIndex = torch.nonzero(self.DPPedge_index_list[-1][0] == GivenNode).squeeze()
        tempList = self.DPPedge_index_list[-1][1,tempIndex].tolist()
        if isinstance(tempList, int):
          LstNegNodesList.extend([tempList])
        else:
          LstNegNodesList.extend(tempList)
      
        # comuSim = self.CosSimilarity(ComuWeight,LstNegNodesCom)
        NodeSim = torch.cdist(x[AjdIndex], x[LstNegNodesList], p=2)

        NodeSim = self.CosSimilarity(x[AjdIndex], x[LstNegNodesList])

        simValue, _ = torch.max(NodeSim, 1)

        lastIndex = self.LayerDppIndex[-1][GivenNode]
        ThreshIndex = []
        TempIndex = (simValue>0.9).nonzero().squeeze().tolist()
        if isinstance(TempIndex, int):
          ThreshIndex.extend([TempIndex])
        else:
          ThreshIndex.extend(TempIndex)
        UnionIndex = list(set(lastIndex).union(set(ThreshIndex)))
        # leftAmount = int(len(index)/2)
        # leftindex = random.sample(index, leftAmount)
        V = EigVec.clone()
        for j in UnionIndex:
            k = torch.argmax(torch.abs(V[j, :]))
            V -= 0.9 * torch.outer(V[:, k] / V[j, k], V[j, :])

        return V

    def CosSimilarity(self,a,b):
        AnormTemp = a.norm(dim=1)[:, None]
        AoneTemp = torch.ones_like(AnormTemp)
        AnormTemp = torch.where(AnormTemp == 0,AoneTemp,AnormTemp)
        Aweight_norm = a / AnormTemp
        BnormTemp = b.norm(dim=1)[:, None]
        BoneTemp = torch.ones_like(BnormTemp)
        BnormTemp = torch.where(BnormTemp == 0,BoneTemp,BnormTemp)
        Bweight_norm = b / BnormTemp
        cosSim = torch.mm(Aweight_norm, Bweight_norm.transpose(0,1))
        return cosSim

import random
def MAD(x,num_nodes):
  MADList =[]
  for i in range (10):
    randindex = random.sample(range(0, num_nodes), int(num_nodes/100))
    rand_x = x[randindex]
    x_norm = rand_x / rand_x.norm(dim=1)[:, None]
    dist = 1-torch.mm(x_norm, x_norm.transpose(0,1))
    one1 = torch.ones_like(dist)
    zreo1 = torch.zeros_like(dist)
    dist_1 = torch.where(dist > 0, one1, zreo1)
    D = dist.sum(0)/dist_1.sum(0)
    one = torch.ones_like(D)
    zero = torch.zeros_like(D)
    D_1 = torch.where(D > 0, one, zero)
    Mad = D.sum()/D_1.sum()
    # print(Mad)
    MADList.append(Mad.detach().numpy())
  return np.average(np.array(MADList))

def GetNodeOverate(edge_index_list,SelCenNodeList):
  overate = []
  for j in range(1,len(edge_index_list)):
    count = 0
    for node in SelCenNodeList:
      EdgeIndex1 = torch.nonzero(edge_index_list[j-1][0]==node)
      EdgeIndex1 = torch.reshape(EdgeIndex1, (-1,))
      EndNode1 = edge_index_list[j-1][1][EdgeIndex1]

      EdgeIndex2 = torch.nonzero(edge_index_list[j][0]==node)
      EdgeIndex2 = torch.reshape(EdgeIndex2, (-1,))
      EndNode2 = edge_index_list[j][1][EdgeIndex2]

      for n in EndNode1:
        count = count + int(torch.isin(n,EndNode2))
    overate.append(count/edge_index_list[j].shape[1])
  print("Layer Node overate: ", np.average(np.array(overate)))
  return np.average(np.array(overate))

def ComOverate(edge_index_list,LComuMatrix,SelCenNodeList):
  overate = []
  for j in range(1,len(edge_index_list)):
    count = 0
    for node in (SelCenNodeList):
      EdgeIndex1 = torch.nonzero(edge_index_list[j-1][0]==node)
      EdgeIndex1 = torch.reshape(EdgeIndex1, (-1,))
      EndNode1 = edge_index_list[j-1][1][EdgeIndex1]

      EdgeIndex2 = torch.nonzero(edge_index_list[j][0]==node)
      EdgeIndex2 = torch.reshape(EdgeIndex2, (-1,))
      EndNode2 = edge_index_list[j][1][EdgeIndex2]

      A1com = torch.nonzero(LComuMatrix[:,EndNode1])[:,0]
      A2com = torch.nonzero(LComuMatrix[:,EndNode2])[:,0]
      for n in A1com:
        count = count + int(torch.isin(n,A2com))
    # print("Layer: %1d/%1d  Comunity Repetition rate: %.4f" % (l-1,l, count/DPPedge_index_list[l].shape[1]))
    overate.append(count/DPPedge_index_list[j].shape[1])
  print("Layer Community overate: ", np.average(np.array(overate)))
  return np.average(np.array(overate))

def ClusterOverate(edge_index_list,ClusterList,SelCenNodeList):
  overate = []
  for j in range(1,len(edge_index_list)):
    count = 0
    for node in (SelCenNodeList):
      EdgeIndex1 = torch.nonzero(edge_index_list[j-1][0]==node)
      EdgeIndex1 = torch.reshape(EdgeIndex1, (-1,))
      EndNode1 = edge_index_list[j-1][1][EdgeIndex1]

      EdgeIndex2 = torch.nonzero(edge_index_list[j][0]==node)
      EdgeIndex2 = torch.reshape(EdgeIndex2, (-1,))
      EndNode2 = edge_index_list[j][1][EdgeIndex2]

      A1com = ClusterList[j-1][EndNode1]
      A2com = ClusterList[j-1][EndNode2]
      for n in A1com:
        count = count + int(torch.isin(n,A2com))
    overate.append(count/DPPedge_index_list[j].shape[1])
  print("Layer Cluste overate:", np.average(np.array(overate)))
  return np.average(np.array(overate))

import sys

def test(Adjcandidate,DPPedge_index_list,train_neg_rate):
      model.eval()
      out,_,_ = model(PyTSubGraph.x, edge_index,Adjcandidate,DPPedge_index_list,train_neg_rate)
      pred = out.argmax(dim=1)  # Use the class with highest probability.

      accs = []
      for mask in [PyTSubGraph.train_mask, PyTSubGraph.val_mask, PyTSubGraph.test_mask]:
          accs.append(int((pred[mask] == PyTSubGraph.y[mask]).sum()) / int(mask.sum()))
      return accs

#----------------
# Set training parameters
num_layers = args.num_layers
Runs = args.runs+1
Epochs = args.epochs+1
# sel_num = 10
#--------------------
#Set whether NegRate is a trainable parameter
#if False:
#   NegRate = 1
train_neg_rate = False
#-----------------

lavgNodeOverateList = []
lavgComOverateList = []
lavgClsOverateList = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PyTSubGraph = PyTSubGraph.to(device)
edge_index = edge_index.to(device)


for num in range(1,Runs):
  seed = num 
  model = LDGCN(seed = seed,
          device = device,
          hidden_channels=256,
          num_layers=num_layers,
          num_nodes=num_nodes,
          DegreeList = DegreeList,
          LabelComu=LabelComu,
          LComuMatrix=LComuMatrix).to(device)


  # print("num_layers",num_layers)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
  criterion = torch.nn.CrossEntropyLoss()
  begin = 0
  if num == 1 :
    print(model)
    # begin = 25

  EpochList = []
  LossList = []
  TrainAccList = []
  ValAccList = []
  TestAccList = []
  MADList =[]

  best_val_acc = final_test_acc = 0
  # print(f'Dataset: {DatasetName}')
  print(f'Number of negative sample: {len(SelCenNodeList)}')
  print(f'Seed: {seed}')
  for epoch in range(begin+1, Epochs):


      start = time.time()
      DPPedge_index_list = []
      ClusterList = []
      model.train()
      optimizer.zero_grad()  # Clear gradients.

      # #Get candidates
      if epoch % 1 ==0:
        candidates = GetCandidateNode(SPEndNodeDict,SelCenNodeList,5)
        Adjcandidate = GetCandiSet(SelCenNodeList,candidates)
      
      out,DPPedge_index_list,ClusterList = model(PyTSubGraph.x, edge_index,Adjcandidate,DPPedge_index_list,train_neg_rate)  
      loss = criterion(out[PyTSubGraph.train_mask], PyTSubGraph.y[PyTSubGraph.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()
      end = time.time()

      train_acc, val_acc, tmp_test_acc = test(Adjcandidate,DPPedge_index_list,train_neg_rate)
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          test_acc = tmp_test_acc
          
      EpochList.append(epoch)
      LossList.append(loss.cpu().detach().numpy())
      TrainAccList.append(train_acc)
      ValAccList.append(val_acc)
      TestAccList.append(test_acc)
      madvalues = MAD(out.cpu(),num_nodes)
      MADList.append(madvalues)

      if epoch == 1 or epoch == begin+1:
          print("---------------")
          print("DPPedge_index_list",DPPedge_index_list[0].shape)
          print("Epoch time:",end-start)
          print("---------------")
          # lavgNodeOverate  = GetNodeOverate(DPPedge_index_list,SelCenNodeList)
          # lavgNodeOverateList.append(lavgNodeOverate)
          print("---------------")
          # lavgComOverate = ComOverate(DPPedge_index_list,LComuMatrix,SelCenNodeList)
          # lavgComOverateList.append(lavgComOverate)
          # print("---------------")  
          # lavgClsOverate = ClusterOverate(DPPedge_index_list,ClusterList,SelCenNodeList)
          # lavgClsOverateList.append(lavgClsOverate)
          # print("---------------")  
            
      if epoch % 5 ==0:
          print(f'num: {num:03d}, Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, MAD: {madvalues:.4f}, Time: {end-start:.4f}')
  if args.save:
    os.makedirs(os.path.join("Result", "ogbn-arxiv"), exist_ok=True)
    PATH = os.path.join("Result", "ogbn-arxiv", "R"+str(args.sample_rate*100)+"%Ogbn-arxiv-LDGCN-Layer-"+str(num_layers)+"-Run-"+str(num)+"-Epoch-"+str(epoch)+".txt")
    savenp = np.array([EpochList,LossList,TrainAccList,ValAccList,TestAccList,MADList])
    np.savetxt(PATH,np.array(savenp),fmt='%f',delimiter=',')
    print("Save Epoch, Path=", PATH)

