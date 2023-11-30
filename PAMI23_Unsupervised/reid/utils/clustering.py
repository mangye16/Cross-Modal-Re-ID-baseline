from re import U
from PIL.Image import new
import numpy
from numpy.lib.arraysetops import unique
import torch
from sklearn.cluster.dbscan_ import dbscan
import math
import faiss
from reid.utils.faiss_utils import search_index_pytorch, search_raw_array_pytorch, \
                            index_init_gpu, index_init_cpu

res = faiss.StandardGpuResources()
res.setDefaultNullStreamAllDevices()

def cal_distance(a, b):
    sum = 0.0
    for i in range(len(a)):
        sum += math.sqrt((a[i] - b[i]) * (a[i] - b[i]))
    return sum


def cluster_label(new_features, new_cams):
    from reid.utils.faiss_rerank import faiss_compute_jaccard_dist
    new_ca_features = []
    new_ir_features = []
    ca_idx_to_full_idx = []
    ir_idx_to_full_idx = []
    for i, item in enumerate(new_features):
        if new_cams[i] in [0, 1, 3, 4]: # for sysu
        # if new_cams[i] in [0]: # for regDB
            new_ca_features.append(item)
            ca_idx_to_full_idx.append(i)
        elif new_cams[i] in [2, 5]: # for sysu
        # elif new_cams[i] in [1]: # for regDB
            new_ir_features.append(item)
            ir_idx_to_full_idx.append(i)

    W_ca = faiss_compute_jaccard_dist(torch.from_numpy(numpy.array(new_ca_features)))
    W_ir = faiss_compute_jaccard_dist(torch.from_numpy(numpy.array(new_ir_features)))
    _, updated_ca_label = dbscan(W_ca, eps=0.5, min_samples=4, metric='precomputed', n_jobs=8) # eps=0.3 for regdb, 0.5 for sysu
    _, updated_ir_label = dbscan(W_ir, eps=0.5, min_samples=4, metric='precomputed', n_jobs=8) # eps=0.3 for regdb, 0.5 for sysu
    # TODO 
    for i, item in enumerate(updated_ir_label):
        if item != -1:
            updated_ir_label[i] += len(numpy.unique(updated_ca_label)) - 1

    ca_center_idx = {}
    ca_label_to_center = {}
    ca_center_features = []
    ca_center_to_label = []
    for i in numpy.unique(updated_ca_label):
        idx = numpy.where(updated_ca_label == i)[0]
        new_center_features = numpy.mean(numpy.array(new_ca_features)[idx], axis=0)
        ca_label_to_center[i] = len(ca_center_features)
        ca_center_to_label.append(i)
        ca_center_features.append(new_center_features)
        ca_center_idx[i] = idx

    ir_center_idx = {}
    ir_center_features = []
    ir_label_to_center = {}
    ir_center_to_label = []
    for i in numpy.unique(updated_ir_label):
        idx = numpy.where(updated_ir_label == i)[0]
        new_center_features = numpy.mean(numpy.array(new_ir_features)[idx], axis=0)
        ir_label_to_center[i] = len(ir_center_features)
        ir_center_to_label.append(i)
        ir_center_features.append(new_center_features)
        ir_center_idx[i] = idx
    
    cnt = 0
    ca_len = numpy.unique(updated_ca_label)
    ir_len = numpy.unique(updated_ir_label)

    for i in ca_len:    
        if i == -1:
            continue
        tmp = torch.unsqueeze(torch.from_numpy(numpy.array(ca_center_features[ca_label_to_center[i]])), dim=0)
        # _, initial_rank = search_raw_array_pytorch(res, torch.from_numpy(numpy.array(ir_center_features)), tmp, 1)
        # initial_rank = initial_rank.cpu().numpy()
        tmp_ir_simi = torch.mm(tmp, torch.from_numpy(numpy.array(ir_center_features)).T)
        initial_rank = torch.topk(tmp_ir_simi, k=1)[1]
        for j in initial_rank[0]:
            tmp_j = torch.unsqueeze(torch.from_numpy(numpy.array(ir_center_features[j])), dim=0)
            # _, initial_rank_j = search_raw_array_pytorch(res, torch.from_numpy(numpy.array(ca_center_features)), tmp_j, 1)
            # initial_rank_j = initial_rank_j.cpu().numpy()
            tmpj_ca_simi = torch.mm(tmp_j, torch.from_numpy(numpy.array(ca_center_features)).T)
            initial_rank_j = torch.topk(tmpj_ca_simi, k=1)[1]
            if ca_label_to_center[i] in initial_rank_j[0]:
                cnt += 1
                idx = ir_center_idx[ir_center_to_label[j]]
                for item in idx:
                    updated_ir_label[item] = i
    print(str(cnt) + " classes in IR classes have been renewed to RGB")
    updated_label = numpy.append(updated_ca_label, updated_ir_label)    
    for i, item in enumerate(updated_ca_label):
        updated_label[ca_idx_to_full_idx[i]] = item
    for i, item in enumerate(updated_ir_label):
        updated_label[ir_idx_to_full_idx[i]] = item
    
    return len(updated_label[updated_label >= 0 ]), updated_label