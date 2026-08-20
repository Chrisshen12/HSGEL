import torch
#print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.metrics import classification_report
from dataset import load_Roman, load_Amazon, load_Cora, load_Reddit, load_Ogb_a, load_Ogb_p, load_Ogb_m, load_Citeseer
from model import GCN,FastGCN,EnsemblePred,GCN_model,GraphEnsemblePred, NodeWiseEnsemble,ResidualEnsemble, TwoStageEnsemble, GCN_full,GateMLP, GraphCorrelationEnsemble, GroupEnsemble
from custom_loader_batchsampler import PredefinedGroupBatchSampler, SeedOrder, StratifiedDegreeSampler
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, ClusterData
from custom_cluster_loader import ClusterLoader
from torch_geometric.index import index2ptr, ptr2index
from torch_sparse import SparseTensor
from torch_geometric.utils import scatter, to_scipy_sparse_matrix, add_self_loops, is_undirected,to_dense_adj, degree
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import PowerNorm
from sklearn.manifold import TSNE
from scipy.stats import norm
from sklearn.metrics import roc_auc_score,f1_score,confusion_matrix,precision_score,recall_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
#from spherecluster import SphericalKMeans
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import seaborn as sns
from torch.nn.functional import cosine_similarity
import random
import copy
import math
import heapq
from collections import defaultdict, Counter
import time
import os
import itertools
from openpyxl import Workbook
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def run_batch_val(args):
	# ----------------------------
	# Setup & Run
	# ----------------------------

	if args.dataset == 'Roman':
		dataset= load_Roman(seed=args.seed)
	elif args.dataset == 'Cora':
		dataset = load_Cora(seed=args.seed)
	elif args.dataset == 'Citeseer':
		dataset = load_Citeseer(seed=args.seed)
	elif args.dataset == 'Amazon':
		dataset= load_Amazon(seed=args.seed)
	elif args.dataset == 'Reddit':
		dataset= load_Reddit(seed=args.seed)
	elif args.dataset == 'Ogba':
		dataset= load_Ogb_a(seed=args.seed)
	elif args.dataset == 'Ogbp':
		dataset= load_Ogb_p(seed=args.seed)
	elif args.dataset == 'Ogbm':
		dataset= load_Ogb_m(seed=args.seed)


	#save_dir = "GCN_1000_50"
	#save_dir = "SAGE_1000_50"
	#os.makedirs(save_dir, exist_ok=True)
	train_mask = dataset.train_mask
	val_mask = dataset.val_mask
	test_mask = dataset.test_mask

	device = args.device
	#device_wrap = torch.device(device)
	x = dataset.x
	num_nodes = x.shape[0]
	#print(sequences[0])
	in_dim = dataset.x.shape[1]
	hidden_dim = args.hidden
	num_classes = dataset.y.unique().shape[0]
	if args.dataset == 'Ogbm':
		dataset.y = dataset.y.long()
		num_classes = 172
	#print(num_classes)
	neighbors = [args.neighbors]*args.hop
	#neighbors = [10,10]
	# if args.dataset=='Ogb':
	#   dataset.y = dataset.y.view(-1)
	#mamba_in = seq_features_out.shape[-1]
	data = args.dataset
	mult=args.batch_size
	batch_num = int(1/mult)
	train_node = int(train_mask.sum())
	val_node = int(val_mask.sum())
	test_node = int(test_mask.sum())
	batch_size_train = math.ceil(mult * train_node)
	batch_size_val = math.ceil(mult * val_node)
	batch_size_test = math.ceil(mult * test_node)
	batch_size_in = math.ceil(mult*num_nodes)
	train_ids = torch.nonzero(train_mask, as_tuple=False).view(-1)
	deg = torch.bincount(dataset.edge_index[0], minlength=num_nodes)
	val_ids = torch.nonzero(val_mask, as_tuple=False).view(-1)
	test_ids = torch.nonzero(test_mask, as_tuple=False).view(-1)

	train_loader = NeighborLoader(dataset, num_neighbors=neighbors, batch_size=batch_size_train, input_nodes=train_mask,shuffle=True)
	val_loader = NeighborLoader(dataset, num_neighbors=neighbors, batch_size=batch_size_val,input_nodes=val_mask,shuffle=False)
	test_loader = NeighborLoader(dataset, num_neighbors=neighbors, batch_size=batch_size_test,input_nodes=test_mask,shuffle=False)



	model = GCN(in_dim=in_dim,hidden=args.hidden,dropout=args.dropout,layer_s=args.layer_size,out_dim=num_classes).to(device)



	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	criterion = nn.CrossEntropyLoss()
	# print(f'Peak GPU Mem: {torch.cuda.max_memory_allocated(device_wrap) / 1024**2:.2f} MB')
	best_val = 0
	patience = 0
	total_epochs_run = 0
	total_train_time = 0.0
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)
	torch.cuda.synchronize()
	K = args.topk  # or 10
	best_models = []
	best_val_history = [] 
	all_val_history = []
	print('training random batch...')
	if args.inference:
		for epoch in range(args.epochs):
			#print(f"epoch:{epoch}----------")
			torch.cuda.reset_peak_memory_stats(device)
			#start_event.record()
			#start_time = time.time()
			model.train()
			# print(f"after model.train: Peak GPU Mem: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
			total_loss = 0
			batch_node = 0
			batch_edge = 0
			total_epochs_run += 1
			#correct=0
			#total=0
			start_event.record()
			counter = 0
			for i,batch in enumerate(train_loader):
				#if i in arr:

				batch = batch.to(device)
				batch_size = batch.batch_size
				optimizer.zero_grad()

				out = model(batch)

				train_loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
				train_loss.backward()
				optimizer.step()
			#   with torch.no_grad():
			#       pred = out[:batch_size].argmax(dim=1)
			#       target = batch.y[:batch_size]
			#       correct += (pred == target).sum().item()
			#       total += batch_size


			# acc = correct / total
			# print(acc)

			end_event.record()
			torch.cuda.synchronize()
			elapsed_time = start_event.elapsed_time(end_event) / 1000.0
			total_train_time += elapsed_time
			#end_time = time.time()
			model.eval()
			correct = 0
			total = 0
			with torch.no_grad():
				for batch in val_loader:
					batch = batch.to(device)
					batch_size = batch.batch_size
					out = model(batch)
					pred = out[:batch_size].argmax(dim=1)
					target = batch.y[:batch_size]
					correct += (pred == target).sum().item()
					total += batch_size


			val = correct/total
			all_val_history.append((total_epochs_run, val))
			#val = (torch.max(predict[val_mask],dim=1)[1] == selected_graph.y[val_mask]).float().mean()
			# ---------- TOP-K MODEL SAVING (replace old saving) ----------

			# add current model
			entry = (val, total_epochs_run, copy.deepcopy(model.state_dict()))
			#print(type(val))
			#heap_updated = False
			if len(best_models) < K:
				heapq.heappush(best_models, entry)
				#print('best model', type(best_models[0][0]))
				#heap_updated = True
			else:
				# compare with worst (min val in heap)
				if val > best_models[0][0]:
					heapq.heapreplace(best_models, entry)
					#heap_updated = True
			if val>best_val:

				#model_state = copy.deepcopy(model.state_dict())
				best_val = val
				best_val_history.append((best_val, total_epochs_run))
				patience =0
			else:
				patience+=1
			if patience>=args.patience:
				break
			#print(f"Val Acc: {val:.4f}, patience: {patience}, Time: {elapsed_time:.4f}s, Peak GPU Mem: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
			#print(f"Val Acc: {val:.4f}, patience: {patience}, Time: {end_time - start_time:.4f}s, Peak GPU Mem: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
			#print(f"Val Acc: {val:.4f}, patience: {patience}")
	print(f"Total epochs run: {total_epochs_run}")
	print(f"Total training time: {total_train_time:.2f} seconds")
	# print(f"Average time per epoch: {total_train_time / total_epochs_run:.2f} seconds")

	best_models_sorted = sorted(best_models, key=lambda x: x[0], reverse=True)

	best_states = [state for _, _, state in best_models_sorted]

	save_dir = "top_val_models"
	os.makedirs(save_dir, exist_ok=True)
	if args.inference:
		for i, state in enumerate(best_states):
			save_path = os.path.join(save_dir, f"{args.dataset}_{args.batch_size}_hop{args.hop}_neighbors{args.neighbors}_GCN_model_topk_{i}.pth")

			torch.save(state, save_path)

	selected_models = args.top_model
	best_states = [torch.load(os.path.join(save_dir,f"{args.dataset}_{args.batch_size}_hop{args.hop}_neighbors{args.neighbors}_GCN_model_topk_{i}.pth"))for i in range(selected_models)]

	single_model_accs = []
	all_logits = []   # <-- store logits instead of argmax predictions
	targets = []
	for m, state in enumerate(best_states):
		model.load_state_dict(state)
		model.eval()

		logits = []
		labels = []

		#for batch in fixed_batches:
		with torch.no_grad():
			for batch in test_loader:
				batch = batch.to(device)
				out = model(batch)

				batch_size = batch.batch_size
				logits.append(out[:batch_size])
				labels.append(batch.y[:batch_size])

			# batch_mask = batch.test_mask
			# logits.append(out[batch_mask].cpu())
			# labels.append(batch.y[batch_mask].cpu())

		logits = torch.cat(logits)        # shape: [N, num_classes]
		labels = torch.cat(labels)
		pred = logits.argmax(dim=1)
		acc = (pred == labels).float().mean().item()
		single_model_accs.append(acc)

		#print(f"Model {m}: Test Acc = {acc:.4f}")
		all_logits.append(logits)
		#print(targets)
		targets = labels                  # same for all models
	# ---- Simple mean ensemble ----
	print(f'max acc: {np.max(single_model_accs):.4f}')
	print(f'avg acc: {np.mean(single_model_accs):.4f}')
	#print(f'avg error: {np.std(single_model_accs):.4f}')
	# ---- Simple mean ensemble ----
	mean_logits = torch.stack(all_logits, dim=0).mean(dim=0)
	pred_ensemble = mean_logits.argmax(dim=1)
	y_true = targets.cpu().int().numpy()
	y_pred = pred_ensemble.cpu().int().numpy()

	# ---- Accuracies ----
	acc_ensemble = (pred_ensemble == targets).float().mean().item()

	print(f"Soft Ensembled Test Acc (avg preds): {acc_ensemble:.4f}")
	preds = torch.stack([l.argmax(dim=1) for l in all_logits], dim=0)
	preds_i = preds[:, None, :]  # [num_models, 1, num_nodes]
	preds_j = preds[None, :, :]  # [1, num_models, num_nodes]

	# Indicator: 1 if model i and model j disagree on node n, 0 otherwise
	indicator = (preds_i != preds_j).float()
	pairwise_disagreement = indicator.mean(dim=2)
	num_models = pairwise_disagreement.shape[0]
	dis_values = pairwise_disagreement.cpu().numpy()
	upper_tri_indices = np.triu_indices(num_models, k=1)
	dis_upper = dis_values[upper_tri_indices]
	mean_dis = dis_upper.mean()
	std_dis = dis_upper.std()
	print(f'avg error: {np.std(single_model_accs):.4f}, disagree mean: {mean_dis:.4f}, disagree std: {std_dis:.4f}')
	
	return mean_logits,targets

def run_batch_val_efficient(args):
	# ----------------------------
	# Setup & Run
	# ----------------------------

	if args.dataset == 'Roman':
		dataset= load_Roman(seed=args.seed)
	elif args.dataset == 'Cora':
		dataset = load_Cora(seed=args.seed)
	elif args.dataset == 'Citeseer':
		dataset = load_Citeseer(seed=args.seed)
	elif args.dataset == 'Amazon':
		dataset= load_Amazon(seed=args.seed)
	elif args.dataset == 'Reddit':
		dataset= load_Reddit(seed=args.seed)
	elif args.dataset == 'Ogba':
		dataset= load_Ogb_a(seed=args.seed)
	elif args.dataset == 'Ogbp':
		dataset= load_Ogb_p(seed=args.seed)
	elif args.dataset == 'Ogbm':
		dataset= load_Ogb_m(seed=args.seed)
	elif args.dataset == 'Fake':
		dataset = load_Reddit_fake(seed=args.seed, avg_degree=args.degree)

	#save_dir = "GCN_1000_50"
	#save_dir = "SAGE_1000_50"
	#os.makedirs(save_dir, exist_ok=True)
	train_mask = dataset.train_mask
	val_mask = dataset.val_mask
	test_mask = dataset.test_mask

	device = args.device
	#device_wrap = torch.device(device)
	x = dataset.x
	num_nodes = x.shape[0]
	#print(sequences[0])
	in_dim = dataset.x.shape[1]
	hidden_dim = args.hidden
	num_classes = dataset.y.unique().shape[0]
	if args.dataset == 'Ogbm':
		dataset.y = dataset.y.long()
		num_classes = 172
	#print(num_classes)
	neighbors = [args.neighbors]*args.hop
	#neighbors = [10,10]
	# if args.dataset=='Ogb':
	#   dataset.y = dataset.y.view(-1)
	#mamba_in = seq_features_out.shape[-1]
	data = args.dataset
	mult=args.batch_size
	batch_num = int(1/mult)
	train_node = int(train_mask.sum())
	val_node = int(val_mask.sum())
	test_node = int(test_mask.sum())
	batch_size_train = math.ceil(mult * train_node)
	batch_size_val = math.ceil(mult * val_node)
	batch_size_test = math.ceil(mult * test_node)
	batch_size_in = math.ceil(mult*num_nodes)
	train_ids = torch.nonzero(train_mask, as_tuple=False).view(-1)
	deg = torch.bincount(dataset.edge_index[0], minlength=num_nodes)
	val_ids = torch.nonzero(val_mask, as_tuple=False).view(-1)
	test_ids = torch.nonzero(test_mask, as_tuple=False).view(-1)

	
	train_loader = NeighborLoader(dataset, num_neighbors=neighbors, batch_size=batch_size_train, input_nodes=train_mask,shuffle=True)
	val_loader = NeighborLoader(dataset, num_neighbors=neighbors, batch_size=batch_size_val,input_nodes=val_mask,shuffle=False)
	test_loader = NeighborLoader(dataset, num_neighbors=neighbors, batch_size=batch_size_test,input_nodes=test_mask,shuffle=False)

	neighbors2 = [args.neighbors2]*args.hop2
	hop_loader = NeighborLoader(dataset, num_neighbors=neighbors2, batch_size=batch_size_train, input_nodes=train_mask,shuffle=True)

	model = GCN(in_dim=in_dim,hidden=args.hidden,dropout=args.dropout,layer_s=args.layer_size,out_dim=num_classes).to(device)



	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	criterion = nn.CrossEntropyLoss()
	# print(f'Peak GPU Mem: {torch.cuda.max_memory_allocated(device_wrap) / 1024**2:.2f} MB')
	best_val = 0
	patience = 0



	total_epochs_run = 0
	total_train_time = 0.0
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)
	torch.cuda.synchronize()
	K = args.topk  # or 10
	lambda_div = 0.05   # diversity strength
	best_val = 0.0
	best_val_acc = 0.0

	patience_counter = 0          # counts epochs without improvement
	plateau_counter = 0           # counts how many plateau escapes attempted
	max_plateau_trials = 1
	cycle_length = args.cycle              # 5 epochs each phase
	cycle_epoch_counter = 0
	current_phase = 0
	plateau_mode = True
	best_models = []
	frozen_best_models = []
	best_val_history = [] 
	all_val_history = []
	diversity_history = []
	save_dir = "top_val_models"
	#best_states = [torch.load(os.path.join(save_dir,f"{args.dataset}_{args.batch_size}_GCN_model_topk_{i}.pth"))for i in range(args.topk)]
	if args.inference:
		best_states = [torch.load(os.path.join(save_dir,f"{args.dataset}_{args.batch_size}_hop{args.hop}_neighbors{args.neighbors}_GCN_model_topk_{i}.pth"))for i in range(5)]
		#best_states = [torch.load(os.path.join(save_dir,f"{args.dataset}_{args.batch_size}_hop{args.hop}_neighbors{args.neighbors}_SAGE_model_topk_{i}.pth"))for i in range(args.topk)]
		#best_states = [torch.load(os.path.join(save_dir,f"{args.dataset}_{args.batch_size}_hop{args.hop}_neighbors{args.neighbors}_GAT_model_topk_{i}.pth"))for i in range(args.topk)]
		best_models = []
		val_counter = -args.topk-1
		for i, state in enumerate(best_states):
			model.load_state_dict(state)
			model.eval()

			correct = 0
			total = 0

			with torch.no_grad():
				for batch in val_loader:
					batch = batch.to(device)
					out = model(batch)
					pred = out[:batch.batch_size].argmax(dim=1)
					correct += (pred == batch.y[:batch.batch_size]).sum().item()
					total += batch.batch_size

			val = correct / total

			entry = (val, val_counter, copy.deepcopy(state))  # epoch=0 since loaded
			heapq.heappush(best_models, entry)
			val_counter += 1
		#print(f"Loaded Model {10}")
		# for i, (val, _, _) in enumerate(best_models):
		#   print(f"Rank {i+1}: {val:.4f}")
		model.load_state_dict(best_states[0])
	print('training random batch...')
	if args.inference:
		for epoch in range(args.epochs):
			if plateau_mode:
				if current_phase == 0:
					current_loader = train_loader
				else:
					current_loader = hop_loader
			else:
				current_loader = train_loader
			#print(f"epoch:{epoch}----------")
			torch.cuda.reset_peak_memory_stats(device)
			#start_event.record()
			#start_time = time.time()
			model.train()
			# print(f"after model.train: Peak GPU Mem: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
			total_loss = 0
			batch_node = 0
			batch_edge = 0
			total_epochs_run += 1
			#correct=0
			#total=0
			start_event.record()
			counter = 0
			for i,batch in enumerate(current_loader):
				#if i in arr:

				batch = batch.to(device)
				batch_size = batch.batch_size
				optimizer.zero_grad()

				out = model(batch)
				train_loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
				train_loss.backward()
				optimizer.step()

			
			end_event.record()
			torch.cuda.synchronize()
			elapsed_time = start_event.elapsed_time(end_event) / 1000.0
			total_train_time += elapsed_time
			#end_time = time.time()
			model.eval()
			correct = 0
			total = 0
			with torch.no_grad():
				for batch in val_loader:
					batch = batch.to(device)
					batch_size = batch.batch_size
					out = model(batch)
					pred = out[:batch_size].argmax(dim=1)
					target = batch.y[:batch_size]
					correct += (pred == target).sum().item()
					total += batch_size


			val = correct/total
			all_val_history.append((total_epochs_run, val))
			#val = (torch.max(predict[val_mask],dim=1)[1] == selected_graph.y[val_mask]).float().mean()
			# ---------- TOP-K MODEL SAVING (replace old saving) ----------

			# add current model
			entry = (val, total_epochs_run, copy.deepcopy(model.state_dict()))
			#print(type(val))
			#heap_updated = False
			if len(best_models) < K:
				heapq.heappush(best_models, entry)
				#print('best model', type(best_models[0][0]))
				#heap_updated = True
			else:
				# compare with worst (min val in heap)
				if val > best_models[0][0]:
					heapq.heapreplace(best_models, entry)
					#heap_updated = True

			if val > best_val:
				best_val = val
				best_val_history.append((best_val, total_epochs_run))
				patience_counter = 0
				#print("Validation improved.")
			else:
				patience_counter += 1
				#print(f"No improvement. Patience: {patience_counter}/{args.patience}")
			if plateau_mode:
				cycle_epoch_counter += 1

				if cycle_epoch_counter >= cycle_length:
					cycle_epoch_counter = 0
					current_phase = 1 - current_phase  # toggle 0 <-> 1
			# Plateau detected
			if patience_counter >= args.patience:
				plateau_counter += 1
				patience_counter = 0
				cycle_epoch_counter = 0
				current_phase = 1  # start with degree_loader

				#print("Entering alternating loader mode")
				if plateau_counter >= max_plateau_trials:
					#print("Maximum plateau trials reached. Stopping training.")
					break

	print(f"Total epochs run: {total_epochs_run}")
	print(f"Total training time: {total_train_time:.2f} seconds")
	# print(f"Average time per epoch: {total_train_time / total_epochs_run:.2f} seconds")

	save_path = os.path.join(
		save_dir,
		f"{args.dataset}_{args.batch_size}_ohop{args.hop}_oneighbors{args.neighbors}_hop{args.hop2}_neighbor{args.neighbors2}_cycle{cycle_length}_top{args.topk}_diverse_hop.pt"
		#f"{args.dataset}_{args.batch_size}_ohop{args.hop}_oneighbors{args.neighbors}_hop{args.hop2}_neighbor{args.neighbors2}_cycle{cycle_length}_top{args.topk}_seed{args.seed}_diverse_hop.pt"
		#f"{args.dataset}_{args.batch_size}_ohop{args.hop}_oneighbors{args.neighbors}_hop{args.hop2}_neighbor{args.neighbors2}_cycle{cycle_length}_top{args.topk}_diverse_SAGE_hop.pt"
		#f"{args.dataset}_{args.batch_size}_ohop{args.hop}_oneighbors{args.neighbors}_hop{args.hop2}_neighbor{args.neighbors2}_cycle{cycle_length}_top{args.topk}_diverse_GAT_hop.pt"
	)
	#---------------------------------------save models
	if args.inference:
		best_states = [state for _,_, state in best_models]
		best_models_sorted = sorted(best_models, key=lambda x: x[0], reverse=True)
		combined = [
		{
			"val_acc": val,
			"state_dict": state
		}
		for (val, _, state) in best_models_sorted]
		torch.save({"models": combined}, save_path)
	#-------------------------------------------

	
	#----------------------------------------------------------
	selected_models = args.top_model
	# state_dict_paths = []
	# for i in range(args.topk):
	#   path = f"top_val_models/{args.dataset}_{args.batch_size}_GCN_model_topk_{i}.pth"
	#   state = torch.load(path, map_location=device)
	#   state_dict_paths.append(state)
	checkpoint = torch.load(save_path)

	models_info = checkpoint["models"]
	models_sorted = sorted(models_info, key=lambda x: x["val_acc"], reverse=True)
	top_models = models_sorted[:selected_models]

	best_states = [entry["state_dict"] for entry in top_models]
	#best_states = [torch.load(f"{i}_100_{args.dataset}_dropout_batch_state_dict.pth") for i in range(10)]
	single_model_accs = []
	all_logits = []   # <-- store logits instead of argmax predictions
	targets = []
	for m, state in enumerate(best_states):
		model.load_state_dict(state)
		model.eval()

		logits = []
		labels = []

		#for batch in fixed_batches:
		with torch.no_grad():
			for batch in test_loader:
				batch = batch.to(device)
				out = model(batch)

				batch_size = batch.batch_size
				logits.append(out[:batch_size])
				labels.append(batch.y[:batch_size])

			# batch_mask = batch.test_mask
			# logits.append(out[batch_mask].cpu())
			# labels.append(batch.y[batch_mask].cpu())

		logits = torch.cat(logits)        # shape: [N, num_classes]
		labels = torch.cat(labels)
		pred = logits.argmax(dim=1)
		acc = (pred == labels).float().mean().item()
		single_model_accs.append(acc)

		#print(f"Model {m}: Test Acc = {acc:.4f}")
		all_logits.append(logits)
		#print(targets)
		targets = labels                  # same for all models
	print(f'max acc: {np.max(single_model_accs):.4f}')
	print(f'avg acc: {np.mean(single_model_accs):.4f}')
	# ---- Simple mean ensemble ----
	mean_logits = torch.stack(all_logits, dim=0).mean(dim=0)
	pred_ensemble = mean_logits.argmax(dim=1)
	y_true = targets.cpu().int().numpy()
	y_pred = pred_ensemble.cpu().int().numpy()

	# ---- Accuracies ----
	acc_ensemble = (pred_ensemble == targets).float().mean().item()

	print(f"Soft Ensembled Test Acc (avg preds): {acc_ensemble:.4f}")
	preds = torch.stack([l.argmax(dim=1) for l in all_logits], dim=0)
	preds_i = preds[:, None, :]  # [num_models, 1, num_nodes]
	preds_j = preds[None, :, :]  # [1, num_models, num_nodes]

	# Indicator: 1 if model i and model j disagree on node n, 0 otherwise
	indicator = (preds_i != preds_j).float()
	pairwise_disagreement = indicator.mean(dim=2)
	num_models = pairwise_disagreement.shape[0]
	dis_values = pairwise_disagreement.cpu().numpy()
	upper_tri_indices = np.triu_indices(num_models, k=1)
	dis_upper = dis_values[upper_tri_indices]
	mean_dis = dis_upper.mean()
	std_dis = dis_upper.std()

	print(f'avg error: {np.std(single_model_accs):.4f}, disagree mean: {mean_dis:.4f}, disagree std: {std_dis:.4f}')
	wrong_mask = (pred_ensemble != targets)

	# incorrect node indices
	wrong_idx = wrong_mask.nonzero(as_tuple=True)[0]
	# plt.figure(figsize=(8, 6))
	# sns.heatmap(pairwise_disagreement.cpu().numpy(),  annot=True, annot_kws={"size": 20}, fmt=".2f", cmap="coolwarm", cbar=True)
	# #plt.title("Pairwise Model Disagreement Heatmap")
	# #plt.xlabel("Model Index")
	# #plt.ylabel("Model Index")
	# plt.show()
	return wrong_idx

def weighted_average_models_multipath(args):
	if args.dataset == 'Roman':
		dataset= load_Roman(seed=args.seed)
	elif args.dataset == 'Amazon':
		dataset= load_Amazon(seed=args.seed)
	elif args.dataset == 'Reddit':
		dataset= load_Reddit(seed=args.seed)
	elif args.dataset == 'Ogba':
		dataset= load_Ogb_a(seed=args.seed)
	elif args.dataset == 'Ogbp':
		dataset= load_Ogb_p(seed=args.seed)


	train_mask = dataset.train_mask
	val_mask = dataset.val_mask
	test_mask = dataset.test_mask

	device = args.device
	#device_wrap = torch.device(device)
	x = dataset.x
	num_nodes = x.shape[0]
	#print(sequences[0])
	in_dim = dataset.x.shape[1]
	hidden_dim = args.hidden
	num_classes = dataset.y.unique().shape[0]
	#print(num_classes)
	neighbors = [args.neighbors]*args.hop

	mult=args.batch_size
	batch_num = int(1/mult)
	train_node = int(train_mask.sum())
	val_node = int(val_mask.sum())
	test_node = int(test_mask.sum())
	batch_size_train = math.ceil(mult * train_node)
	batch_size_val = math.ceil(mult * val_node)
	batch_size_test = math.ceil(mult * test_node)
	train_ids = torch.nonzero(train_mask, as_tuple=False).view(-1)
	deg = torch.bincount(dataset.edge_index[0], minlength=num_nodes)
	val_ids = torch.nonzero(val_mask, as_tuple=False).view(-1)
	test_ids = torch.nonzero(test_mask, as_tuple=False).view(-1)


	train_loader = NeighborLoader(dataset, num_neighbors=neighbors, batch_size=batch_size_train, input_nodes=train_mask,shuffle=False)
	#train_loader = NeighborLoader(dataset, num_neighbors=neighbors, batch_size=batch_size_train, input_nodes=train_mask,subgraph_type='bidirectional',shuffle=True)
	val_loader = NeighborLoader(dataset, num_neighbors=neighbors, batch_size=batch_size_val,input_nodes=val_mask,shuffle=False)
	test_loader = NeighborLoader(dataset, num_neighbors=neighbors, batch_size=batch_size_test,input_nodes=test_mask,shuffle=False)


	save_dir = "top_val_models"
	configs = [(4,2),(4,3),(3,3)] #ROMAN best
	#configs = [(4,2),(4,3)] #Amazon best
	#configs = [(3,5),(3,3), (4,3)]
	all_best_states = []
	best_states = []
	seen_vals = set()
	for hop2, neigh2 in configs:
		save_path = os.path.join(
			save_dir,
			f"{args.dataset}_{args.batch_size}"
			f"_ohop{args.hop}_oneighbors{args.neighbors}"
			f"_hop{hop2}_neighbor{neigh2}"
			f"_cycle{args.cycle}_top{args.topk}_diverse_hop.pt"
		)
		# save_path = os.path.join(
		# 	save_dir,f"{args.dataset}_{args.batch_size}_ohop{args.hop}_oneighbors{args.neighbors}_hop{hop2}_neighbor{neigh2}_cycle{args.cycle}_top{args.topk}_frequent_class.pt")
		checkpoint = torch.load(save_path, map_location=device)

		models_info = checkpoint["models"]
		# best_models = [entry["state_dict"] for entry in models_info]

		# all_best_states.extend(best_models)
		for idx, entry in enumerate(models_info):

			val = entry["val_acc"]

			# avoid float precision issue
			val_key = round(val, 4)

			# keep unique validation score
			if val_key not in seen_vals:

				seen_vals.add(val_key)

				best_states.append(entry["state_dict"])
	#best_states = all_best_states
	num_models = len(best_states)
	model = GCN(in_dim=in_dim,hidden=args.hidden,dropout=args.dropout,layer_s=args.layer_size,out_dim=num_classes).to(device)
	def cache_preds(best_states, model, test_loader):
		single_model_accs = []
		all_logits = []   # <-- store logits instead of argmax predictions
		targets = []
		for m, state in enumerate(best_states):
			model.load_state_dict(state)
			model.eval()

			logits = []
			labels = []

			#for batch in fixed_batches:
			with torch.no_grad():
				for batch in test_loader:
					batch = batch.to(device)
					out = model(batch)

					batch_size = batch.batch_size
					logits.append(out[:batch_size])
					labels.append(batch.y[:batch_size])

				# batch_mask = batch.test_mask
				# logits.append(out[batch_mask].cpu())
				# labels.append(batch.y[batch_mask].cpu())

			logits = torch.cat(logits)        # shape: [N, num_classes]
			labels = torch.cat(labels)
			pred = logits.argmax(dim=1)
			acc = (pred == labels).float().mean().item()
			single_model_accs.append(acc)

			#print(f"Model {m}: Test Acc = {acc:.4f}")
			all_logits.append(logits)
			#print(targets)
			targets = labels                  # same for all models
		#print(f'max acc: {np.max(single_model_accs):.4f}')
		#print(f'avg acc: {np.mean(single_model_accs):.4f}')
		# ---- Simple mean ensemble ----
		mean_logits = torch.stack(all_logits, dim=0).mean(dim=0)
		pred_ensemble = mean_logits.argmax(dim=1)
		y_true = targets.cpu().int().numpy()
		y_pred = pred_ensemble.cpu().int().numpy()

		# ---- Accuracies ----
		acc_ensemble = (pred_ensemble == targets).float().mean().item()

		print(f"Soft Ensembled Test Acc (avg preds): {acc_ensemble:.4f}")
		all_preds = torch.stack(all_logits, dim=0)
		batch_preds_list = []

		start = 0
		for batch in test_loader:
			B = batch.batch_size
			end = start + B

			# slice dataset dimension
			batch_preds = all_preds[:, start:end, :]  # [M, B, C]
			batch_preds_list.append(batch_preds)

			start = end

		return batch_preds_list
	#train_preds = cache_preds(best_states, model, train_loader)
	#val_preds = cache_preds(best_states, model, val_loader)
	test_preds = cache_preds(best_states, model, test_loader)
	train_preds = cache_preds(best_states, model, train_loader)
	val_preds = cache_preds(best_states, model, val_loader)
	models = []

	model = CorrelationEnsemble(num_models=num_models,out_dim=num_classes,hidden_dim=args.hidden2).to(device)


	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	criterion = nn.CrossEntropyLoss()
	# print(f'Peak GPU Mem: {torch.cuda.max_memory_allocated(device_wrap) / 1024**2:.2f} MB')
	best_val = 0
	patience = 0
	

	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)
	torch.cuda.synchronize()
	print('training random batch...')
	if args.inference:
		for epoch in range(args.epochs):
			#print(f"epoch:{epoch}----------")
			torch.cuda.reset_peak_memory_stats(device)
			#start_event.record()
			#start_time = time.time()
			model.train()
			# print(f"after model.train: Peak GPU Mem: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
			total_loss = 0
			batch_node = 0
			batch_edge = 0
			correct = 0
			total = 0
			t_loss = []
			start_event.record()
			for i,batch in enumerate(train_loader):
			#for i,batch in enumerate(test_loader):
			#for i,batch in enumerate(fixed_train):

				batch = batch.to(device)
				batch_size = batch.batch_size
				preds = train_preds[i].to(device)
				M, B, C = preds.shape

				# [M, B*C]
				flat_preds = preds.reshape(M, -1)

				# normalize
				flat_preds = F.normalize(flat_preds, p=2, dim=1)

				# [M, M]
				corr_matrix = flat_preds @ flat_preds.T

				# remove self-correlation
				mask = ~torch.eye(M, dtype=torch.bool, device=preds.device)

				mean_corr = (
					corr_matrix.masked_fill(~mask, 0)
					.sum(dim=1)
					/ (M - 1)
				)
				weights = 1.0 - mean_corr
				weights = torch.clamp(weights, min=1e-6)
				weights = mean_corr / mean_corr.sum()
				#preds = test_preds[i].to(device)
				optimizer.zero_grad()

				out,_ = model(preds,weights)
				#print(timing)

				train_loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
				t_loss.append(train_loss.item())
				train_loss.backward()
				optimizer.step()
			# 	pred = out[:batch_size].argmax(dim=1)
			# 	target = batch.y[:batch_size]
			# 	correct += (pred == target).sum().item()
			# 	total += batch_size

			t_loss_mean = torch.tensor(t_loss).mean().item()
			t_loss_std  = torch.tensor(t_loss).std(unbiased=False).item()
			model.eval()
			test_loss = 0
			correct = 0
			total = 0
			
			with torch.no_grad():
				for i, batch in enumerate(test_loader):
				#for i, batch in enumerate(fixed_test):
					batch = batch.to(device)
					preds = test_preds[i].to(device)
					M, B, C = preds.shape

					# [M, B*C]
					flat_preds = preds.reshape(M, -1)

					# normalize
					flat_preds = F.normalize(flat_preds, p=2, dim=1)

					# [M, M]
					corr_matrix = flat_preds @ flat_preds.T

					# remove self-correlation
					mask = ~torch.eye(M, dtype=torch.bool, device=preds.device)

					mean_corr = (
						corr_matrix.masked_fill(~mask, 0)
						.sum(dim=1)
						/ (M - 1)
					)
					#weights = mean_corr / mean_corr.sum()
					weights = 1.0 - mean_corr
					weights = torch.clamp(weights, min=1e-6)
					weights = mean_corr / mean_corr.sum()
					out, w = model(preds,weights)
					#out = preds.mean(dim=0)

					loss = criterion(out[:batch.batch_size],
									 batch.y[:batch.batch_size])
					test_loss += loss.item()

					pred = out[:batch.batch_size].argmax(dim=1)
					correct += (pred == batch.y[:batch.batch_size]).sum().item()
					total += batch.batch_size

			test_loss /= len(test_loader)
			test_acc = correct / total
			# acc = correct / total
			# print(acc)
			end_event.record()
			torch.cuda.synchronize()
			elapsed_time = start_event.elapsed_time(end_event) / 1000.0
			#end_time = time.time()
			
			model.eval()
			correct = 0
			total = 0
			with torch.no_grad():
				for i,batch in enumerate(val_loader):
				#for i,batch in enumerate(test_loader):
				#for i,batch in enumerate(fixed_val):
					batch = batch.to(device)
					batch_size = batch.batch_size
					preds = val_preds[i].to(device)
					M, B, C = preds.shape

					# [M, B*C]
					flat_preds = preds.reshape(M, -1)

					# normalize
					flat_preds = F.normalize(flat_preds, p=2, dim=1)

					# [M, M]
					corr_matrix = flat_preds @ flat_preds.T

					# remove self-correlation
					mask = ~torch.eye(M, dtype=torch.bool, device=preds.device)

					mean_corr = (
						corr_matrix.masked_fill(~mask, 0)
						.sum(dim=1)
						/ (M - 1)
					)
					weights = 1.0 - mean_corr
					weights = torch.clamp(weights, min=1e-6)
					weights = mean_corr / mean_corr.sum()
					#preds = test_preds[i].to(device)
					out,_ = model(preds,weights)
					pred = out[:batch_size].argmax(dim=1)
					target = batch.y[:batch_size]
					correct += (pred == target).sum().item()
					total += batch_size


			val = correct / total
			#val = (torch.max(predict[val_mask],dim=1)[1] == selected_graph.y[val_mask]).float().mean()
			if val>best_val:
				#torch.save(model.state_dict(), f'{args.arr}_100_{args.dataset}_batch_state_dict.pth')
				model_state = copy.deepcopy(model.state_dict())
				best_val = val
				patience =0
			else:
				patience+=1
			if patience>=args.patience:
				break
			#print(f"Val Acc: {val:.4f}, patience: {patience}, Time: {elapsed_time:.4f}s, Peak GPU Mem: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
			#print(f"Val Acc: {val:.4f}, patience: {patience}, Time: {end_time - start_time:.4f}s, Peak GPU Mem: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
			print(f"train loss: {t_loss_mean:.4f}, Val Acc: {val:.4f}, patience: {patience}, test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")
	model.load_state_dict(model_state)
	#model.load_state_dict(torch.load(f'{args.arr}_100_{args.dataset}_batch_state_dict.pth'))
	model.eval()
	with torch.no_grad():
		all_preds = []
		all_targets = []
		correct = 0
		total = 0
		for i,batch in enumerate(test_loader):
		#for i,batch in enumerate(fixed_test):
			batch = batch.to(device)
			batch_size = batch.batch_size
			preds = test_preds[i].to(device)
			#test_edge_index = edge_matrix(batch,args.walk_len)
			out, weight = model(preds)
			#print(weight)
			pred = out[:batch_size].argmax(dim=1)
			target = batch.y[:batch_size]
			all_preds.append(pred.cpu())
			all_targets.append(target.cpu())
			correct += (pred == target).sum().item()
			total += batch_size

		acc = correct / total
	# 	y_true = torch.cat(all_targets).numpy()
	# 	y_pred = torch.cat(all_preds).numpy()
		
	# 	# NEW: Generate the Classification Report
	# 	report = classification_report(y_true, y_pred, zero_division=0)

	#print(weight)
	# print("\n--- Classification Report ---\n")
	# print(report)

	print(f"batch acc: {acc:.4f}")

	return model



def main(args):
	run_batch_val(args)
	run_batch_val_efficient(args)
	weighted_average_models_multipath(args)
if __name__ == "__main__":
	parser = argparse.ArgumentParser("HAN")
	parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='gpu or cpu')
	parser.add_argument('--epochs', type=int, default=3000)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--dropout', type=float, default=0)
	parser.add_argument('--topk', type=int, default=10)
	parser.add_argument('--selected_models', type=int, default=10)
	parser.add_argument('--weight', type=int, default=3)
	parser.add_argument('--subset', type=int, default=5)
	parser.add_argument('--top_model', type=int, default=10)
	parser.add_argument('--cycle', type=int, default=5)
	parser.add_argument('--clusters', type=int, default=5)
	parser.add_argument('--patience',type=int,default=50)
	parser.add_argument('--num_model',type=int,default=3)
	parser.add_argument('--seed',type=int,default=0)
	parser.add_argument('--hidden',type=int,default=64)
	parser.add_argument('--hidden2',type=int,default=64)
	parser.add_argument('--neighbors',type=int,default=10)
	parser.add_argument('--neighbors2',type=int,default=5)
	parser.add_argument('--neighbors3',type=int,default=5)
	parser.add_argument('--batch_size',type=float,default=.5)
	parser.add_argument('--C',type=float,default=.5)
	parser.add_argument('--gamma',type=float,default=.5)
	parser.add_argument('--layer_size',type=int,default=2)
	parser.add_argument('--layer_size2',type=int,default=1)
	parser.add_argument('--choice',type=int,default=0)
	parser.add_argument('--hop', type=int, default=2, help="neighbor hop")
	parser.add_argument('--hop2', type=int, default=3, help="neighbor hop")
	parser.add_argument('--hop3', type=int, default=4, help="neighbor hop")
	parser.add_argument('--alpha', type=float, default=0.8)
	parser.add_argument('--dataset',type=str,default='Roman')
	parser.add_argument('--attention_dim',type=int,default=8)
	parser.add_argument('--inference', default=True, action='store_false', help='Bool type')

	args = parser.parse_args()
	main(args)