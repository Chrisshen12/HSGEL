import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import knn_graph
import math
import pandas as pd


class GCN(torch.nn.Module):
	def __init__(self, in_dim, hidden, dropout, layer_s, out_dim):
		super(GCN, self).__init__()
		self.layers = []
		#self.enc = GATConv(in_dim, hidden)
		self.enc = nn.Linear(in_dim, hidden)
		self.bn = nn.LayerNorm(hidden)
		self.hidden = hidden
		self.heads = 1
		for i in range(layer_s):
			self.layers.append(GCNConv(hidden,hidden))
			#self.layers.append(SAGEConv(hidden,hidden))
			#self.layers.append(GATConv(hidden,hidden))
			#self.layers.append(GATConv(hidden,hidden, heads=self.heads,concat=True))
		#self.head_weight = torch.nn.Parameter(torch.ones(self.heads))
		self.dec = nn.Linear(hidden,out_dim)
		#self.dec = GATConv(hidden,out_dim)

		self.layers = torch.nn.ModuleList(self.layers)
		self.dropout = dropout
		
	def forward(self,batch, return_embedding=False):
		#x = F.dropout(x,self.dropout,training=self.training)
		#print(f'g,{g}')
		#batch = batch.to('cuda:0')
		x = batch.x
		g = batch.edge_index
		x = F.dropout(x,self.dropout,training=self.training)
		#x = F.dropout(x,self.dropout,training=self.training)
		#g, _ = dropout_edge(g,p=self.dropout,force_undirected=True,training=self.training)
		#g, _, _ = dropout_node(g,p=self.dropout,training=self.training)
		#print(f'drop_g,{g}')
		#x0 = self.enc(x)
		#x = self.enc(x)

		#x = F.dropout(x,self.dropout,training=self.training)
		x = F.relu(self.enc(x))
		#x = F.leaky_relu(self.enc(x,g),0.2)
		for i,conv in enumerate(self.layers):
			#x = F.leaky_relu(conv(x,g),0.2)+x
			#x = self.bn(F.relu(conv(x,g))+x)

			#x_ = F.relu(conv(x,g))

			#x_ = x_.view(-1, self.heads, self.hidden)  # [N, heads, hidden_dim]

			# learnable per-head weights (same across all nodes)
			
			#w = torch.softmax(self.head_weight, dim=0)            # normalize weights

			#x = (x_ * w.view(1, self.heads, 1)).sum(dim=1)+x
			#print(x.shape)
			x = F.relu(conv(x,g))+x
			#print(f"After layer {i}:", x.shape, torch.cuda.memory_allocated()/1024**2, "MB")
			#x = conv(x,g)+x0
		#x = F.dropout(x,self.dropout,training=self.training)
		h=x
		#print(x.shape)
		x = self.dec(x)
		if return_embedding:
			return x, h

		return x

class CorrelationEnsemble(nn.Module):
	def __init__(self, num_models, out_dim, hidden_dim):
		super().__init__()

		self.num_models = num_models
		self.out_dim = out_dim

		# -------- Stage 1: Node-level attention (per model) --------
		self.node_attn = nn.Sequential(
			nn.Linear(out_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, out_dim)
		)

		# -------- Stage 2: Global model weights --------
		self.model_weights = nn.Parameter(torch.ones(num_models))

		# interpolation parameter (optional)
		self.alpha = nn.Parameter(torch.tensor(0.1))

	def forward(self, preds,weights):
		"""
		preds: [M, B, C]
		"""

		M, B, C = preds.shape
		avg_pred = (
			preds *
			weights.view(M, 1, 1)
			).sum(dim=0)
		#avg_pred = preds.mean(dim=0)
		weights = torch.softmax(self.model_weights, dim=0)  # [M]

		# reshape for broadcasting
		model_w = weights.view(-1, 1, 1)  # [M,1,1]

		# weighted sum across models
		weighted = (model_w * preds).sum(dim=0) 
		#final_pred = weighted
		#final_pred = avg_pred + torch.tanh(self.alpha) * (weighted - avg_pred)
		final_pred = weighted + torch.tanh(self.alpha) * (weighted - avg_pred)

		# ===== Stage 1: Node-wise attention per model =====
		# reshape to apply attention
		# preds_flat = preds.reshape(M * B, C)

		# attn_scores = self.node_attn(preds_flat)
		# attn_scores = torch.softmax(attn_scores, dim=1)

		# refined = attn_scores * preds_flat
		# refined = refined.reshape(M, B, C)

		# # ===== Stage 2: Model-level weighting =====
		# model_w = torch.softmax(self.model_weights, dim=0)  # [M]

		# weighted = (model_w.view(M, 1, 1) * refined).sum(dim=0)

		# # optional interpolation with average
		
		# #final_pred = weighted
		# final_pred = avg_pred + torch.tanh(self.alpha) * (weighted - avg_pred)
		#final_pred = avg_pred + torch.tanh(self.alpha) * weighted

		return final_pred, model_w

