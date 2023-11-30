import torch
import numpy as np
import os.path as osp
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.cluster._dbscan import dbscan
from sklearn.cluster import KMeans
from reid.utils.rerank import compute_jaccard_dist
from reid.utils.faiss_rerank import faiss_compute_jaccard_dist
from reid.utils.clustering import cluster_label
import scipy.io as sio 
torch.autograd.set_detect_anomaly(True)


def img_association(network, propagate_loader, min_sample=4, eps=0,
                    rerank=False, k1=20, k2=6, intra_id_reinitialize=False):

    network.eval()
    print('Start Inference...')
    features = []
    global_labels = []
    real_labels = []
    file_names = []
    all_cams = []

    with torch.no_grad():
        for c, data in enumerate(propagate_loader):
            images = data[0]
            r_label = data[2]
            g_label = data[3]
            f_name = data[1] 
            cam = data[4]
            images = images.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            embed_feat = network(images)
            features.append(embed_feat.cpu())
            real_labels.append(r_label)
            global_labels.append(g_label)
            all_cams.append(cam)
            file_names.append(np.array(f_name))

    features = torch.cat(features, dim=0).numpy()
    real_labels = torch.cat(real_labels, dim=0).numpy()
    global_labels = torch.cat(global_labels, dim=0).numpy()
    all_cams = torch.cat(all_cams, dim=0).numpy()
    # file_names = torch.cat(file_names, dim=0).numpy()
    print('  features: shape= {}'.format(features.shape))
    
    # if needed, average camera-style transferred image features
    new_features = []
    new_cams = []
  
    for glab in np.unique(global_labels):
        idx = np.where(global_labels == glab)[0]
        new_features.append(np.mean(features[idx], axis=0))
        new_cams.append(all_cams[idx])

    new_features = np.array(new_features)
    new_cams = np.array(new_cams).squeeze()
    del features, all_cams
    new_features = new_features / np.linalg.norm(new_features, axis=1, keepdims=True)  # l2-normalize

    # self-similarity for association
    print('perform image grouping...')
    _, updated_label = cluster_label(new_features, new_cams)
    print('  eps in cluster: {:.3f}'.format(eps))
    print('  updated_label: num_class= {}, {}/{} images are associated.'
          .format(updated_label.max() + 1, len(updated_label[updated_label >= 0]), len(updated_label)))

    if intra_id_reinitialize:
        print('re-computing initialized intra-ID feature...')
        intra_id_features = []
        intra_id_labels = []
        for cc in np.unique(new_cams):
            percam_ind = np.where(new_cams == cc)[0]
            percam_feature = new_features[percam_ind, :]
            percam_label = updated_label[percam_ind]
            percam_class_num = len(np.unique(percam_label[percam_label >= 0]))
            percam_id_feature = np.zeros((percam_class_num, percam_feature.shape[1]), dtype=np.float32)
            cnt = 0
            for lbl in np.unique(percam_label):
                if lbl >= 0:
                    ind = np.where(percam_label == lbl)[0]
                    id_feat = np.mean(percam_feature[ind], axis=0)
                    percam_id_feature[cnt, :] = id_feat
                    intra_id_labels.append(lbl)
                    cnt += 1
            percam_id_feature = percam_id_feature / np.linalg.norm(percam_id_feature, axis=1, keepdims=True)
            intra_id_features.append(torch.from_numpy(percam_id_feature))
        return updated_label, intra_id_features

