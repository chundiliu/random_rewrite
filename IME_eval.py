from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import sys

from gae.utils import mask_test_edges, preprocess_graph_sp, preprocess_graph, get_roc_score, get_roc_score_matrix, mask_test_rows, sparse_mx_to_torch_sparse_tensor, neg_sample
from gae.preprocess_graph import *
from joblib import Parallel, delayed
import ipdb
import ctypes
from sklearn.manifold import TSNE

import tensorflow as tf
from IME_layer import compute_IME, use_original_feature

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=99999, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=2048, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=2048, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.003, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size.')

#AE, VAE, AE_batch, VAE_batch
mode = "AE"
print("mode is " + mode)
args = parser.parse_args()
for key in vars(args):
    print(key + ":" + str(vars(args)[key]))

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)

def batch_sparse_dense_matmul(sparse_matrix, dense_matrix, num_batch = 10):
    batch_size = int(np.ceil((float)(sparse_matrix.get_shape()[0].value) / num_batch))
    sparse_res = []
    for i in range(num_batch):
        start = i * batch_size
        end = min((i + 1) * batch_size, sparse_matrix.get_shape()[0].value)
        sparse_res.append(tf.sparse_tensor_dense_matmul(
                tf.sparse_slice(sparse_matrix, [start, 0], [end - start, sparse_matrix.get_shape()[1].value]), dense_matrix))
    return tf.concat(sparse_res, axis=0)




def gae_for(args, position):
    print("Using {} dataset".format(args.dataset_str))
    #qhashes, chashes = load_hashes()
    Q, X = load_data()
    prebuild = "/media/chundi/3b6b0f74-0ac7-42c7-b76b-00c65f5b3673/revisitop/cnnimageretrieval-pytorch/data/test/matlab_data/GEM_wDis_prebuild.bin"
    Q_features = "/media/chundi/3b6b0f74-0ac7-42c7-b76b-00c65f5b3673/revisitop/cnnimageretrieval-pytorch/data/test/matlab_data/roxford5k_GEM_lw_query_feats.npy" #"/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/landmarks/oxfordRe/evaluation/roxHD_query_fused.npy"
    X_features = "/media/chundi/3b6b0f74-0ac7-42c7-b76b-00c65f5b3673/revisitop/cnnimageretrieval-pytorch/data/test/matlab_data/roxford5k_GEM_index.npy"
    D_features = "/media/chundi/3b6b0f74-0ac7-42c7-b76b-00c65f5b3673/revisitop/cnnimageretrieval-pytorch/data/test/matlab_data/roxford5k_GEM_Dis.npy"
    Q = Q.T
    X = X.T
    #ipdb.set_trace()
    data, IME_params = compute_IME(features=np.concatenate([Q, X], axis=0))
    np.save('IME_params', IME_params)
    np.save('IME_index', data)
    # IME_params = np.load('IME_params.npy').tolist()
    # data = np.load('IME_index.npy')
    data, _ = compute_IME(X, params=IME_params)
    print(data.shape)
    rankings = []
    for i in range(70):
        qq = Q[i]
        qq, _ = compute_IME(qq, params=IME_params)
        revop_inner_prod = np.matmul(data, qq)
        revop_preds = np.argsort(-revop_inner_prod, axis=0)
        rankings.append(revop_preds)
        sys.stdout.write("\r" + "evaluating queries: [" + str(i) + "/" + str(70) + "]")
        sys.stdout.flush()
    rankings = np.array(rankings)
    rankings = np.reshape(rankings, (rankings.shape[0], rankings.shape[1]))
    rankings = rankings.T
    revop_map = eval_revop(rankings, silent=True)
    print("IME map:{}".format(revop_map))

    ipdb.set_trace()



    


if __name__ == '__main__':
    num_runs = 50
    #temp = Parallel(n_jobs=34)(delayed(gae_for)(args, i) for i in range(num_runs))
    gae_for(args, 0)

    score = [temp[i][0] for i in range(len(temp))]
    val_score = [temp[i][1] for i in range(len(temp))]
    val_best_roc_revop = [temp[i][2] for i in range(len(temp))]
    val_best_ap_revop = [temp[i][3] for i in range(len(temp))]
    print("layer=2")
    print("dim=512 + 128")
    print("k=5")
    print("best_revop:" +  str(np.mean(score)))
    print("best_val_revop:" + str(np.mean(val_score)))
    print("val_best_roc_revop:" + str(np.mean(val_best_roc_revop)))
    print("val_best_ap_revop" + str(np.mean(val_best_ap_revop)))
    #print(score)
    #print(val_score)
    #print(np.mean(score))
    #print(np.mean(val_score))
    #print(np.var(score))
    #print(np.var(val_score))

    #for i in range(20):
    #    score.append(gae_for(args))

