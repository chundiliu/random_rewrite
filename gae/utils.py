import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from revop import *
from heapq import *
import sys

def EGT(adj, Q_end, n, threshold=0.5):
    r = np.zeros((Q_end, adj.shape[1]))
    for i in range(Q_end):
        count = 0
        q = []
        added = {}
        for j in range(adj.shape[1]):
            if j > Q_end:
                # negate the similarity to get max heap
                heappush(q, (-adj[i,j], j))

        while count < n and len(q) != 0:
            value, index = heappop(q)
            if index not in added:
                r[i, count] = index - Q_end
                count += 1
                added[index] = True
                for j in range(adj.shape[1]):
                    if j > Q_end:
                        heappush(q, (-adj[index,j], j))
            if not len(q) != 0:
                peak_value = q[0][0]
                while not len(q) != 0 and peak_value > threshold:
                    value, index = heappop(q)
                    if index not in added:
                        r[i, count] = index - Q_end
                        count += 1
                        added[index] = True
                        for j in range(adj.shape[1]):
                            if j > Q_end:
                                heappush(q, (-adj[index,j], j))
                    if not len(q) != 0:
                        peak_value = q[0][0]
        for j in range(adj.shape[1]):
            if j not in added:
                r[i, count] = j
                count += 1
        sys.stdout.write("\r" + "Doing EGT [" + str(i) + "/" + str(Q_end) + "]")
        sys.stdout.flush()
    print(r.shape)
    eval_revop(r.T)


def neg_sample(pos_edges, total_length, size, step):
    a = np.random.permutation(total_length)
    neg_edges = []
    count = 0
    for i in a:
        if i not in pos_edges:
            neg_edges.append(i)
            count += 1
        if count == size:
            break
    sys.stdout.write("\r neg_sampling: " + str(step))
    sys.stdout.flush()
    return neg_edges

def get_roc_score_matrix(emb, Q_end=70):
    def qe_vec(preds,Q,X, k = 2):
        Qexp = np.array([(np.sum(X[:,top[:k]],axis=1)+query)/(k+1) for query,top in zip(Q.T,preds.T)]).T
        B = Qexp[:, 70:]
        eval_revop(np.argsort(-np.matmul(B.T,Qexp),axis=0))
        return np.matmul(X.T,Qexp), Qexp.T

    k = 2
    embQ = emb[:Q_end,:].T
    embX = emb[Q_end:,:].T
    #f = np.concatenate((embQ.T,embX.T))
    #sim = np.matmul(f,f.T)
    #EGT(sim, Q_end, 500)
    #sim_top = np.argpartition(sim,-k,1)[:,-k:]
    #sim_qe, f = qe_vec(sim_top.T,f.T,f.T, 3)

    revop_inner_prod = np.matmul(embX.T,embQ)
    revop_preds = np.argsort(-revop_inner_prod,axis=0)
    revop_map = eval_revop(revop_preds,silent=True)
    return revop_map


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        with open("gae/data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
        # objects.append(
        #     pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "gae/data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def shuffle_adj(adj, features):
    # properly shuffle the adj matrix
    # if row1 and row3 switched, then the columns1 and columns3 should be switched too?
    #for i in range(2000):
    #    sys.stdout.write("\r shuffling for the validation sampling: [" + str(i) + "/" + str(2000) + "]")
    #    r1, r2 = np.random.randint(0, adj.shape[0], size=2)
    #    adj[[r1,r2], :] = adj[[r2,r1], :]
    #    adj[:, [r1,r2]] = adj[:, [r2,r1]]
    #    features[[r1,r2], :] = features[[r2,r1], :]
    np.random.seed(20190222)
    rand = np.random.permutation(adj.shape[0])
    adj = adj[rand, :][:, rand]
    features = features[rand, :]
    return adj, features

def mask_test_rows(adj, features):
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    #assert np.diag(adj.todense()).sum() == 0
    #rows, columns = adj.nonzero()
    ## make symmetric
    #for i in range(rows.shape[0]):
    #    adj[columns[i], rows[i]] = adj[rows[i], columns[i]] if adj[columns[i], rows[i]] == 0 else adj[columns[i], rows[i]]
    #for r in rows:
    #    for c in columns:
    #        print(adj[r,c])
    #adj = sp.triu(adj)
    #adj = adj + adj.T


    # sample rows to be train valid and test
    num_rows = adj.shape[0]
    num_train = int(adj.shape[0] - np.floor(adj.shape[0] / 10.))
    adj, features = shuffle_adj(adj, features)
    # try to take validation as bottom
    adj_train = adj[:num_train, :num_train]
    adj_valid = adj[num_train:, :]
    #adj_train = adj[num_val:, num_val:]
    #adj_train = adj_train + adj_train.T

    features_valid = features[num_train:, :]
    features_train = features[:num_train, :]

    return adj_train, adj_valid, features_train, features_valid


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def preprocess_graph_sp(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

def preprocess_graph(adj):
    #adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    # adj_normalized = adj_normalized * adj_normalized * adj_normalized
    
    
    #return  torch.from_numpy(adj_normalized.todense()).float()
    #return torch.from_numpy(sparse_to_tuple(adj_normalized))
    return sp.csr_matrix(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(edges_pos)), np.zeros(len(edges_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score
