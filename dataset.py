import torch
import torch.nn as nn
from torch_geometric.datasets import HeterophilousGraphDataset, Planetoid, Reddit2
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected, degree,dropout_edge
import random
from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from torch.serialization import safe_globals
from torch_geometric.data.data import DataEdgeAttr,DataTensorAttr
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from torch_geometric.data.storage import GlobalStorage

import warnings
warnings.filterwarnings("ignore")


def load_Roman(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	dataset = HeterophilousGraphDataset(root='tmp/Roman', name='Roman-empire')
	train_mask = dataset.train_mask[:,0]
	val_mask = dataset.val_mask[:,0]
	test_mask = dataset.test_mask[:,0]
	data = dataset[0]
	data.edge_index = to_undirected(dataset.edge_index)

	data.train_mask = train_mask
	data.val_mask = val_mask
	data.test_mask = test_mask

	return data

def load_Amazon(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	dataset = HeterophilousGraphDataset(root='tmp/Amazon', name='Amazon-ratings')
	train_mask = dataset.train_mask[:,0]
	val_mask = dataset.val_mask[:,0]
	test_mask = dataset.test_mask[:,0]
	data = dataset[0]
	data.edge_index = to_undirected(dataset.edge_index)

	data.train_mask = train_mask
	data.val_mask = val_mask
	data.test_mask = test_mask
	#print(data.edge_index.shape)
	#print(compute_homophily(data.edge_index,data.y).mean())
	return data


def load_Reddit(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)

	dataset = Reddit2(root='tmp/Reddit2')
	data = dataset[0]
	#print(data)
	#print(data.y.unique().numel())
	data.edge_index = to_undirected(dataset.edge_index)
	#print(compute_homophily(data.edge_index,data.y).nanmean())
	
	# deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
	# plt.figure(figsize=(6, 4))
	# plt.hist(deg.cpu().numpy(), bins=50, edgecolor='black', alpha=0.7)
	# #plt.title(f"Node Degree Distribution\nMean={mean_deg:.2f}, Std={std_deg:.2f}")
	# plt.xlabel("Degree")
	# plt.ylabel("Count")
	# plt.grid(alpha=0.3)
	# plt.tight_layout()
	# plt.show()
	# mean_deg = deg.mean().item()
	# std_deg = deg.std().item()
	# print(std_deg)
	# # Compute statistics
	# avg_deg = deg.mean().item()
	# print(avg_deg)
	#print(data.edge_index.shape)
	return data


def load_Cora(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	# ----------------------------
	# Load Cora dataset
	# ----------------------------
	#dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
	dataset = Planetoid(root='/tmp/Cora', name='Cora')
	#dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
	data = dataset[0]
	#print(data)
	#print(data.y.unique().numel())
	train_mask = dataset.train_mask
	val_mask = dataset.val_mask
	test_mask = dataset.test_mask
	data.edge_index = to_undirected(dataset.edge_index)


	return data

def load_Citeseer(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	# ----------------------------
	# Load Cora dataset
	# ----------------------------
	#dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
	#dataset = Planetoid(root='/tmp/Cora', name='Cora')
	dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
	data = dataset[0]
	#print(data)
	#print(data.y.unique().numel())
	train_mask = dataset.train_mask
	val_mask = dataset.val_mask
	test_mask = dataset.test_mask
	data.edge_index = to_undirected(dataset.edge_index)


	return data

def load_Ogb_a(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	#dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="tmp/Ogb")
	with safe_globals([DataEdgeAttr,DataTensorAttr,GlobalStorage]):
		#dataset = PygNodePropPredDataset(name="ogbn-products", root="tmp/Ogbp")
		dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="tmp/Ogba")

	# Get the graph object (PyTorch Geometric Data object)
	data = dataset[0]

	data.edge_index = to_undirected(dataset.edge_index)
	#print("Number of nodes:", data.num_nodes)
	#print("Number of edges:", data.num_edges)
	split_idx = dataset.get_idx_split()

	# Load split indices for train/val/test
	num_nodes = data.num_nodes
	train_mask = torch.zeros(num_nodes, dtype=torch.bool)
	val_mask = torch.zeros(num_nodes, dtype=torch.bool)
	test_mask = torch.zeros(num_nodes, dtype=torch.bool)

	train_mask[split_idx["train"]] = True
	val_mask[split_idx["valid"]] = True
	test_mask[split_idx["test"]] = True

	data.train_mask = train_mask
	data.val_mask = val_mask
	data.test_mask = test_mask
	#print(data.x.shape)
	data.y = dataset.y.view(-1)

	return data

def load_Ogb_p(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	#dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="tmp/Ogb")
	with safe_globals([DataEdgeAttr,DataTensorAttr,GlobalStorage]):
		dataset = PygNodePropPredDataset(name="ogbn-products", root="tmp/Ogbp")

	# Get the graph object (PyTorch Geometric Data object)
	data = dataset[0]

	data.edge_index = to_undirected(dataset.edge_index)
	#print("Number of nodes:", data.num_nodes)
	#print("Number of edges:", data.num_edges)
	split_idx = dataset.get_idx_split()

	# Load split indices for train/val/test
	num_nodes = data.num_nodes
	train_mask = torch.zeros(num_nodes, dtype=torch.bool)
	val_mask = torch.zeros(num_nodes, dtype=torch.bool)
	test_mask = torch.zeros(num_nodes, dtype=torch.bool)

	train_mask[split_idx["train"]] = True
	val_mask[split_idx["valid"]] = True
	test_mask[split_idx["test"]] = True

	data.train_mask = train_mask
	data.val_mask = val_mask
	data.test_mask = test_mask
	#print(data.x.shape)
	data.y = dataset.y.view(-1)
	#print(compute_homophily(data.edge_index,data.y).nanmean())
	#print(data.train_mask)
	# deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
	# avg_degree = deg.mean().item()
	# print(avg_degree)
	#print(y.unique().numel())
	# print(train_mask.sum())
	# print(val_mask.sum())
	# print(test_mask.sum())
	#tokens = hop_to_token(data,k_hops=1)
	#print(data)
	return data

def load_Ogb_m(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	#dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="tmp/Ogb")
	# with safe_globals([DataEdgeAttr,DataTensorAttr,GlobalStorage,np.core.multiarray.scalar,np.dtype,
	# np.ndarray,
	# np.int64,
	# np.float32,
	# np.dtypes.Int64DType]):
	# 	#dataset = PygNodePropPredDataset(name="ogbn-products", root="tmp/Ogbp")
	# 	dataset = PygNodePropPredDataset(name="ogbn-papers100M", root="tmp/Ogbm")

	# # #Get the graph object (PyTorch Geometric Data object)
	# data = dataset[0]
	# print(data)
	# data.edge_index, _ = dropout_edge(data.edge_index, p = .5)
	# data.edge_index = to_undirected(data.edge_index, data.num_nodes)
	# #data.edge_index = to_undirected(dataset.edge_index)
	# #data.edge_index = dataset.edge_index
	# #print("Number of nodes:", data.num_nodes)
	# #print("Number of edges:", data.num_edges)
	# split_idx = dataset.get_idx_split()

	# # Load split indices for train/val/test
	# num_nodes = data.num_nodes
	# train_mask = torch.zeros(num_nodes, dtype=torch.bool)
	# val_mask = torch.zeros(num_nodes, dtype=torch.bool)
	# test_mask = torch.zeros(num_nodes, dtype=torch.bool)

	# train_mask[split_idx["train"]] = True
	# val_mask[split_idx["valid"]] = True
	# test_mask[split_idx["test"]] = True

	# data.train_mask = train_mask
	# data.val_mask = val_mask
	# data.test_mask = test_mask
	# #print(data.x.shape)
	# data.y = dataset.y.view(-1)
	# #print(data)
	# torch.save(data, "Ogbm_graph_drop0.5.pt")
	with safe_globals([DataEdgeAttr,DataTensorAttr,GlobalStorage,np.core.multiarray.scalar,np.dtype,
	np.ndarray,
	np.int64,
	np.float32,
	np.dtypes.Int64DType]):
		data = torch.load("Ogbm_graph_drop0.5.pt")
	#print(compute_homophily(data.edge_index,data.y).nanmean())
	#print(data.train_mask)
	# deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
	# avg_degree = deg.mean().item()
	# print(avg_degree)
	#print(y.unique().numel())
	# print(train_mask.sum())
	# print(val_mask.sum())
	# print(test_mask.sum())
	#tokens = hop_to_token(data,k_hops=1)
	#print(data)
	return data
