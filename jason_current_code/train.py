from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
import sys
from torch import optim

from gae.model import GCNModelVAE, GCNModelAE, GCNModelVAE_batch, GCNModelAE_batch
from gae.optimizer import loss_function, SGLD, pSGLD, loss_function_ae
from gae.utils import mask_test_edges, preprocess_graph_sp, preprocess_graph, get_roc_score, get_roc_score_matrix, mask_test_rows, sparse_mx_to_torch_sparse_tensor, neg_sample
from gae.preprocess_graph import *
from joblib import Parallel, delayed
import ctypes
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=512, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=128, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.0003, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')

#AE, VAE, AE_batch, VAE_batch
mode = "AE_batch"
print("mode is " + mode)
args = parser.parse_args()
for key in vars(args):
    print(key + ":" + str(vars(args)[key]))

def gae_for(args, position):
    print("Using {} dataset".format(args.dataset_str))
    #qhashes, chashes = load_hashes()
    Q, X = load_data()
    prebuild = "/media/gcn-gae/GEM_wDis_prebuild.bin"
    Q_features = "/media/gcn-gae/roxford5k_GEM_lw_query_feats.npy" #"/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/landmarks/oxfordRe/evaluation/roxHD_query_fused.npy"
    X_features = "/media/gcn-gae/roxford5k_GEM_index.npy"
    D_features = "/media/gcn-gae/roxford5k_GEM_Dis.npy"
    #adj, features, adj_Q, features_Q = load_from_prebuild(prebuild, Q_features, X_features, D_features, k=5)
    #cut_size = 800000
    #adj = adj[:cut_size, :cut_size]
    #adj_Q = adj_Q[:, :cut_size]
    #features = features[:cut_size]
    #Q = np.load("/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/landmarks/oxfordRe/evaluation/roxHD_query_fused.npy").T.astype(np.float32)
    #X = np.load("/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/landmarks/oxfordRe/evaluation/roxHD_index_fused.npy").T.astype(np.float32)
    #D = np.load("/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/landmarks/revisitop1m/revisitDistractors_fused_3s_cq.npy").T.astype(np.float32)
    #X = np.concatenate((X.T,D.T)).T
    # load the distractor too, shape should be (2048, 1M)
    adj, features = gen_graph_index(Q, X, k=5, k_qe=3, do_qe=False)

    adj_Q, features_Q = gen_graph(Q, X, k=5, k_qe=3, do_qe=False) #generate validation/revop evaluation the same way as training
    features_all = np.concatenate([features_Q, features])
    features_all = torch.from_numpy(features_all)
    #adj_Q = adj_Q.todense()
    #adj_all = np.concatenate([adj_Q, adj.todense()])
    #adj_all = np.pad(adj_all, [[0,0], [Q.shape[1], 0]], "constant")
    adj_all = sp.vstack((adj_Q, adj))
    zeros = sp.csr_matrix((adj_all.shape[0], Q.shape[1]))
    adj_all = sp.hstack((zeros, adj_all))
    adj_all = sp.csr_matrix(adj_all)
    rows, columns = adj_all.nonzero()
    print("Making Symmetry")
    for i in range(rows.shape[0]):
        if rows[i] < Q.shape[1]:
            adj_all[columns[i], rows[i]] = adj_all[rows[i], columns[i]]
        else:
            break
    #adj_all = sp.csr_matrix(adj_all)
    print("preprocessing adj_all")
    adj_all_norm = preprocess_graph(adj_all)
    #adj = add_neighbours_neighbour(adj)
    #adj1, features1 = load_data(args.dataset_str)
    features = torch.from_numpy(features)
    #features_all = torch.from_numpy(features_all)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj = adj_orig

    #adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    print("Sampling validation")
    adj_train, adj_val, features, features_valid = mask_test_rows(adj, features)
    adj = adj_train

    # Some preprocessing
    print("preprocessing adj")
    adj_norm = preprocess_graph(adj)
    #adj_norm_label = preprocess_graph_sp(adj)

    adj_label = adj_train + sp.eye(adj_train.shape[0]) #adj_norm_label + sp.eye(adj_train.shape[0]) #adj_train + sp.eye(adj_train.shape[0])
    #rows, columns = adj_label.nonzero()
    #adj_label[columns, rows] = adj_label[rows, columns]
    # adj_label = sparse_to_tuple(adj_label)
    #adj_label = torch.FloatTensor(adj_label.toarray())

    print("adj sum: " + str(adj.sum()))
    pos_weight = float(float(adj.shape[0]) * adj.shape[0] - adj.sum()) / adj.sum()
    pos_weight = 0
    print("top part: " + str(float(float(adj.shape[0]) * adj.shape[0] - adj.sum())))
    print("pos wieght: " + str(pos_weight))
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # for validation data processing:
    zero = sp.csr_matrix((adj_train.shape[0], adj_val.shape[0]))
    adj_train_ext = sp.hstack((zero, adj_train))
    adj_evaluate = sp.vstack((adj_val, adj_train_ext))
    adj_evaluate = sp.csr_matrix(adj_evaluate)
    rows, columns = adj_evaluate.nonzero()
    val_edges = []
    val_edges_false = []
    pos = {}
    print("getting positive edges")
    all_val = [i for i in range(len(rows)) if rows[i] < adj_val.shape[0]]
    for i in all_val:
        sys.stdout.write("\r sampling edges for validtion: [" + str(i) + "]")
        val_edges.append((rows[i], columns[i]))
        if rows[i] not in pos:
            pos[rows[i]] = []
        pos[rows[i]].append(columns[i])
    #for i in range(rows.shape[0]):
    #    sys.stdout.write("\r sampling edges for validtion: [" + str(i) + "/" + str(adj_val.shape[0]) + "]")
    #    sys.stdout.flush()
    #    if rows[i] < adj_val.shape[0]:
    #        val_edges.append((rows[i], columns[i]))
    #        if rows[i] not in pos:
    #            pos[rows[i]] = []
    #        pos[rows[i]].append(columns[i])
    #        adj_evaluate[columns[i], rows[i]] = adj_evaluate[rows[i], columns[i]]
    #    else:
    #        break
    step = 0
    neg_per_pos = 100
    #for r in pos:
    #    p = pos[r]
    #neg_edges = Parallel(n_jobs=40)(delayed(neg_sample)(pos[i], adj_val.shape[1], neg_per_pos, i) for i in pos)
    #val_edges_false = [(i, item) for i in range(len(neg_edges)) for item in neg_edges[i]]
        #a = np.random.permutation(adj_val.shape[1])
        #a = [i for i in a if i not in p]
        #a = a[:100]
        #for i in a:
        #    val_edges_false.append((r, i))
        ##count = 0
        ##i = 0
        #sys.stdout.write("\r sampling neg edges for validtion: [" + str(step) + "/" + str(len(pos)) + "]")
        #sys.stdout.flush()
        #step += 1
        #while count < 100:
        #    if a[i] not in p:
        #        val_edges_false.append((r, a[i]))
        #        count += 1
        #    i += 1

    print("preprocessing adj_evaluate")
    adj_evaluate_norm = preprocess_graph(adj_evaluate)
    #adj_evaluate_norm_label = preprocess_graph_sp(adj_evaluate)

    adj_label_evaluate = adj_evaluate + sp.eye(adj_evaluate.shape[0]) #adj_evaluate_norm_label+ sp.eye(adj_evaluate.shape[0]) #adj_evaluate + sp.eye(adj_evaluate.shape[0])
    #adj_label_evaluate = torch.FloatTensor(adj_label_evaluate.toarray()) #sparse_mx_to_torch_sparse_tensor(adj_label_evaluate)
    features_evaluate = np.concatenate([features, features_valid])
    features_evaluate = torch.from_numpy(features_evaluate)
    # validation done 

    if mode == "VAE":
        model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
        adj_label = torch.FloatTensor(adj_label.toarray())
        adj_label_evaluate = torch.FloatTensor(adj_label_evaluate.toarray())
    elif mode == "AE":
        model = GCNModelAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
        adj_label = torch.FloatTensor(adj_label.toarray())
        adj_label_evaluate = torch.FloatTensor(adj_label_evaluate.toarray())
    elif mode == "VAE_batch":
        model = GCNModelVAE_batch(feat_dim, args.hidden1, args.hidden2, args.dropout)
    elif mode == "AE_batch":
        model = GCNModelAE_batch(feat_dim, args.hidden1, args.hidden2, args.dropout)
    #model = torch.nn.DataParallel(model)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = pSGLD(model.parameters(), lr=args.lr)
    #optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    hidden_emb = None
    pos_weight = torch.from_numpy(np.array(pos_weight))
    t = time.time()
    best = 0
    best_epoch = 0
    best_val_cost = 99999
    best_val_epoch = 0
    best_val_epoch_revop = 0
    prev_loss = 0
    prev_val_loss = 99999
    best_val_roc = 0
    best_val_ap = 0
    best_val_roc_revop = 0
    best_val_ap_revop = 0
    torch.set_num_threads(128)
    print("NUM THREADS USED")
    print(torch.get_num_threads())
    for epoch in range(args.epochs):
        #t = time.time()
        model.train()
        optimizer.zero_grad()
        #print(type(features))
        #print(type(adj_norm.coalesce().indices()))
        #print(features.shape)
        #print(type(adj_norm))
        #just_adj = sparse_mx_to_torch_sparse_tensor(adj)
        #recovered, mu, logvar = model(features, just_adj.coalesce().indices(), edge_weight=just_adj.coalesce().values())
        #print(adj_norm.shape)
        #print(features.shape)
        #print(adj_label.shape)

        # need to do some sampling here                         shape: adj_norm [4557, 4557], features [4557, 2048], adj_label [4557, 4557]
        # maybe sample all the non_zero adj and 1k negative
        # print(adj_label)
        #print(adj_norm.coalesce().indices())
        #position = adj_norm.coalesce().indices()
        #d = adj_norm.to_dense().data.numpy()
        #sample_size = 5000
        #column_exclude = set()
        #random_perm = np.random.permutation(d.shape[0])

        if mode == "VAE":
            recovered, mu, logvar = model(features, adj_norm)
            loss = loss_function(preds=recovered, labels=adj_label,
                                 mu=mu, logvar=logvar, n_nodes=n_nodes,
                                 norm=norm, pos_weight=pos_weight)

        elif mode == "VAE_batch":
            recovered, mu, logvar, loss = model(features, adj_norm, labels=adj_label, n_nodes=n_nodes, norm=norm, pos_weight=pos_weight, opt=optimizer, training=True)
            #recovered, mu, logvar = model(features, adj_norm)
            #recovered = recovered[i:i+500]
            #sample_adj_label = sample_adj_label[i:i+500]
            #loss = loss_function(preds=recovered, labels=adj_label,
            #                     mu=mu, logvar=logvar, n_nodes=n_nodes,
            #                     norm=norm, pos_weight=pos_weight)
        elif mode == "AE":
            recovered, mu = model(features, adj_norm)
            loss = loss_function_ae(preds=recovered, labels=adj_label,
                                    norm=norm, pos_weight=pos_weight)
        elif mode == "AE_batch":
            recovered, mu, loss = model(features, adj_norm, labels=adj_label, n_nodes=n_nodes, norm=norm, pos_weight=pos_weight, opt=optimizer, training=True)

        if mode == "AE" or mode == "VAE":
            loss.backward()
            optimizer.step()
        try:
            cur_loss = loss.item()
        except Exception:
            cur_loss = loss
        #optimizer.step()
        # sample rows only: non-square
        #for i in range(0, d.shape[0], sample_size):
        #    selection = random_perm[i:i+sample_size]
        #    to_keep = selection
        #    sample_adj_norm = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(d[to_keep, :]))
        #    sample_adj = adj[to_keep, :]
        #    sample_features = features
        #    recovered, mu, logvar = model(sample_features, sample_adj_norm)
        #    loss = loss_function(preds=recovered, labels=sample_adj_label,
        #                         mu=mu, logvar=logvar, n_nodes=n_nodes,
        #                         norm=norm, pos_weight=pos_weight)
        #    loss.backward()
        #    cur_loss = loss.item()
        #    optimizer.step()
        #    sys.stdout.write("\r                                                                                                                                       ")
        #    sys.stdout.write("\r" + "sampling [" + str(i) + "/" + str(d.shape[0]))
        #    sys.stdout.flush()

        # sample rows + take their postiives and add to rows (make it square)
        #for i in range(0, d.shape[0], sample_size):
        #    selection = random_perm[i:i+sample_size]
        #    to_keep = np.nonzero(d[selection, :])
        #    # (array([0, 1, 2, 2]), array([0, 1, 0, 1]))
        #    the_set = set(list(to_keep[0]) + list(to_keep[1]))
        #    temp = set(list(to_keep[0]))
        #    to_keep = list(the_set - column_exclude)
        #    # column_exclude.union(temp)
        #    # these ar ethe rows and columns that we need ne select
        #    sample_adj_norm = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(d[to_keep, :][:,to_keep]))
        #    sample_features = features[to_keep, :]
        #    sample_adj_label = adj_label[to_keep, :][:,to_keep]
        #    #print(samplei_adj_norm.shape)
        #    #print(sample_features.shape)
        #    #print(sample_adj_label.shape)
        #    #print(sample.shape)
        #    sample_adj = adj[to_keep, :][:,to_keep]
        #    pos_weight = float(sample_adj.shape[0] * sample_adj.shape[0] - sample_adj.sum()) / sample_adj.sum()
        #    pos_weight = torch.from_numpy(np.array(pos_weight))
        #    norm = sample_adj.shape[0] * sample_adj.shape[0] / float((sample_adj.shape[0] * sample_adj.shape[0] - sample_adj.sum()) * 2)

        #    n_nodes, feat_dim = sample_features.shape

        #    if mode == "VAE":
        #        recovered, mu, logvar = model(sample_features, sample_adj_norm)
        #        #recovered = recovered[i:i+500]
        #        #sample_adj_label = sample_adj_label[i:i+500]
        #        loss = loss_function(preds=recovered, labels=sample_adj_label,
        #                             mu=mu, logvar=logvar, n_nodes=n_nodes,
        #                             norm=norm, pos_weight=pos_weight)
        #    elif mode == "AE":
        #        recovered = model(sample_features, sample_adj_norm)
        #        loss = loss_function_ae(preds=recovered, labels=sample_adj_label,
        #                                norm=norm, pos_weight=pos_weight)
        #    loss.backward()
        #    cur_loss = loss.item()
        #    optimizer.step()
        #    sys.stdout.write("\r                                                                                                                                       ")
        #    sys.stdout.write("\r" + "sampling [" + str(i) + "/" + str(d.shape[0]) + "]....size of sample=" + str(len(sample_features)))
        #    sys.stdout.flush()

        sys.stdout.write("\r                                                                                                                                          \r")
        if (epoch + 1) % 1 == 0:
            model.eval()
            #adj_dense = adj_train.todense()
            #adj_val_dense = adj_val.todense()
            #adj_train_ext = np.pad(adj_dense, [[0,0], [adj_val_dense.shape[0], 0]], "constant")
            #adj_evaluate = np.concatenate([adj_val_dense, adj_train_ext])
            #zero = sp.csr_matrix((adj_train.shape[0], adj_val.shape[0]))
            #adj_train_ext = sp.hstack((zero, adj_train))
            #adj_evaluate = sp.vstack((adj_val, adj_train_ext))
            ##zeros = sp.csr_matrix((adj_evaluate.shape[0], adj_val.shape[1]))
            ##adj_evaluate = sp.hstack((zeros, adj_evaluate))
            #adj_evaluate = sp.csr_matrix(adj_evaluate)
            #rows, columns = adj_evaluate.nonzero()
            #for i in range(rows.shape[0]):
            #    if rows[i] < adj_val.shape[1]:
            #        adj_evaluate[columns[i], rows[i]] = adj_evaluate[rows[i], columns[i]]
            #    else:
            #        break

            #adj_evaluate_norm = preprocess_graph(adj_evaluate)

            #adj_label_evaluate = adj_evaluate + sp.eye(adj_evaluate.shape[0])
            #adj_label_evaluate = torch.FloatTensor(adj_label_evaluate.toarray())
            ##adj_label_evaluate = sparse_to_tuple(adj_label_evaluate)

            #features_evaluate = np.concatenate([features, features_valid])
            #features_evaluate = torch.from_numpy(features_evaluate)
            just_adj_evaluate = sparse_mx_to_torch_sparse_tensor(adj_evaluate)
            #recovered, mu, logvar = model(features_evaluate, just_adj_evaluate.coalesce().indices(), just_adj_evaluate.coalesce().values())
            #recovered, mu, logvar = model(features_evaluate, adj_evaluate_norm)
            #val_loss = loss_function(preds=recovered, labels=adj_label_evaluate,
            #                         mu=mu, logvar=logvar, n_nodes=n_nodes,
            #                         norm=norm, pos_weight=pos_weight)
            if mode == "VAE":
                recovered, mu, logvar = model(features_evaluate, adj_evaluate_norm)
                val_loss = loss_function(preds=recovered, labels=adj_label_evaluate,
                                         mu=mu, logvar=logvar, n_nodes=n_nodes,
                                         norm=norm, pos_weight=pos_weight)
            elif mode == "AE":
                recovered, mu = model(features_evaluate, adj_evaluate_norm)
                val_loss = loss_function_ae(preds=recovered, labels=adj_label_evaluate,
                                         norm=norm, pos_weight=pos_weight)
            elif mode == "VAE_batch":
                recovered, mu, logvar, val_loss = model(features_evaluate, adj_evaluate_norm, labels=adj_label_evaluate, n_nodes=n_nodes, norm=norm, pos_weight=pos_weight, opt=None, training=False)
            elif mode == "AE_batch":
                recovered, mu, val_loss = model(features_evaluate, adj_evaluate_norm, labels=adj_label_evaluate, n_nodes=n_nodes, norm=norm, pos_weight=pos_weight, opt=None, training=False)
            val_emb = mu.data.numpy()
            #roc_curr, ap_curr = get_roc_score(val_emb, val_edges, val_edges_false)



            # do one q at a time
            #revop_map = eval_each_q(model, adj_all, features_all, Q.shape[1])

            # hack by appending stuff on top of adj
            if mode == "VAE":
                _, mu, _ = model(features_all, adj_all_norm)
            elif mode == "AE":
                _, mu = model(features_all, adj_all_norm)
            elif mode == "VAE_batch":
                mu = model(features_all, adj_all_norm, None, None, None, None, None, just_mu=True, training=False)
            elif mode == "AE_batch":
                mu = model(features_all, adj_all_norm, None, None, None, None, None, just_mu=True, training=False)
            hidden_emb = mu.data.numpy()
            ## get validation loss
            #recovered, mu, logvar = model(features, adj_norm)
            #val_loss = loss_function(preds=recovered, labels=adj_label,
            #                         mu=mu, logvar=logvar, n_nodes=n_nodes,
            #                         norm=norm, pos_weight=pos_weight)
            revop_map = get_roc_score_matrix(hidden_emb, Q.shape[1])

            if best <= revop_map:
                emb = hidden_emb
                Q_end = Q.shape[1]
                best = revop_map
                best_epoch = epoch + 1
                # write it into a file and do egt on that
                #embQ = emb[:Q_end,:].T
                #embX = emb[Q_end:,:].T
                #np.save("/media/jason/28c9eee1-312e-47d0-88ce-572813ebd6f1/graph/gae-pytorch/best_embedding2.npy",hidden_emb)
                #concat = np.concatenate((embQ.T,embX.T))
                #revop_inner_prod = np.matmul(concat, concat.T)
                #revop_preds = np.argsort(-revop_inner_prod,axis=0)
                #if revop_map > 54:
                #    f = open("best_result.txt", "w")
                #    for i in range(revop_preds.shape[1]):
                #        if i < Q_end:
                #            f.write(qhashes[i] + ",")
                #        else:
                #            f.write(chashes[i - Q_end] + ",")
                #        for j in revop_preds[:,i]:
                #            if j < Q_end:
                #                f.write(qhashes[j] + " " + str(int(revop_inner_prod[j,i] * 1000)) + " ")
                #            else:
                #                f.write(chashes[j - Q_end] + " " + str(int(revop_inner_prod[j,i] * 1000)) + " ")
                #        f.write("\n")
                #        f.flush()
                #        #for j in range()
                #    f.close()

    
            if best_val_cost > val_loss: #prev_val_loss - val_loss > 0 and prev_val_loss - val_loss > prev_loss - cur_loss and best_val_cost > val_loss:
                best_val_cost = val_loss
                best_val_epoch = epoch + 1
                best_val_epoch_revop = revop_map
    
            #if best_val_roc < roc_curr:
            #    best_val_roc = roc_curr
            #    best_val_roc_revop = revop_map

            #if best_val_ap < ap_curr:
            #    best_val_ap = ap_curr
            #    best_val_ap_revop = revop_map
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                  "val_loss=", "{:.5f}".format(val_loss),
                  #"val_roc_curr=", "{:.5f}".format(roc_curr),
                  #"val_ap_curr=", "{:.5f}".format(ap_curr),
                  "revop=", "{:.5f}".format(revop_map),
                  "best_revop=", "{:.5f}".format(best),
                  "revop_at_best_val=", "{:.5f}".format(best_val_epoch_revop),
                  #"revop_at_best_val_roc=", "{:.5f}".format(best_val_roc_revop),
                  #"revop_at_best_ap_roc=", "{:.5f}".format(best_val_ap_revop),
                  "time=", "{:.5f}".format(time.time() - t)
                  )
            prev_val_loss = val_loss
            prev_loss = cur_loss
            t = time.time()
    print("Optimization Finished!")

    #roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    #print('Test ROC score: ' + str(roc_score))
    #print('Test AP score: ' + str(ap_score))
    return best, best_val_epoch_revop, best_val_roc_revop, best_val_ap_revop


def eval_each_q(model, adj_all, features_all, q_length):
    adj_q = adj_all[:q_length, q_length:]
    adj_i = adj_all[q_length:, q_length:]
    features_q = features_all[:q_length, :]
    features_i = features_all[q_length:, :]
    rankings = []
    for i in range(q_length):
        adj_all_evaluation = sp.vstack((adj_q[i, :], adj_i))
        zeros = sp.csr_matrix((adj_all_evaluation.shape[0], 1))
        adj_all_evaluation = sp.hstack((zeros, adj_all_evaluation))
        adj_all_evaluation = sp.csr_matrix(adj_all_evaluation)
        features_all_evaluation = np.concatenate([features_q[i:i+1, :], features_i])
        features_all_evaluation = torch.from_numpy(features_all_evaluation)

        rows, columns = adj_all_evaluation.nonzero()
        for j in range(rows.shape[0]):
            if rows[j] < 1:
                adj_all_evaluation[columns[j], rows[j]] = adj_all_evaluation[rows[j], columns[j]]
            else:
                break
        adj_all_evaluation = preprocess_graph(adj_all_evaluation)

        if mode == "VAE":
            _, mu, _ = model(features_all_evaluation, adj_all_evaluation)
        elif mode == "AE":
            _, mu = model(features_all_evaluation, adj_all_evaluation)
        elif mode == "VAE_batch":
            mu = model(features_all_evaluation, adj_all_evaluation, None, None, None, None, None, just_mu=True, training=False)
        elif mode == "AE_batch":
            mu = model(features_all_evaluation, adj_all_evaluation, None, None, None, None, None, just_mu=True, training=False)
        emb = mu.data.numpy()
        embX = emb[1:, :].T
        embQ = emb[:1, :].T
        revop_inner_prod = np.matmul(embX.T,embQ)
        revop_preds = np.argsort(-revop_inner_prod,axis=0)
        rankings.append(revop_preds)
        sys.stdout.write("\r" + "evaluating queries: [" + str(i) + "/" + str(q_length) + "]")
        sys.stdout.flush()
    rankings = np.array(rankings)
    rankings = np.reshape(rankings, (rankings.shape[0], rankings.shape[1]))
    rankings = rankings.T
    revop_map = eval_revop(rankings,silent=True)
    return revop_map
    


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

