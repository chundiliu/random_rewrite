import numpy as np
import time
import scipy.sparse as sp
import networkx as nx
from revop import *
import sys
import time
from joblib import Parallel, delayed
from multiprocessing import Process, Manager

DATA_PATH = '/media/chundi/3b6b0f74-0ac7-42c7-b76b-00c65f5b3673/revisitop/cnnimageretrieval-pytorch/data/test/matlab_data'
if os.path.exists(DATA_PATH)==False:
    DATA_PATH = '/d2/lmk_code/revisitop/data'
if os.path.exists(DATA_PATH)==False:
    DATA_PATH = '/media/gcn-gae/data'

assert os.path.exists(DATA_PATH),'out of data path to search, add your path to preprocess_graph!'

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def add_neighbours_neighbour(org_top):
    new_org_top = []
    for i in range(len(org_top)):
        temp = [l for l in org_top[i]]
        for j in org_top[i]:
            temp2 = [l for l in org_top[j]]
            temp += temp2
        new_org_top.append(temp)
    return new_org_top


def get_inliers(file_path):
    f = open(file_path, "r")
    lines = [line[:-1] for line in f.readlines()]
    d = {}
    for line in lines:
        parts = line.strip().split(",")
        q = parts[0]
        cAnds = parts[1].split(" ")
        for i in range(0, len(cAnds), 2):
            c = cAnds[i]
            s = int(cAnds[i+1])
            if q not in d:
                d[q] = {}
            if c not in d:
                d[c] = {}
            if c not in d[q]:
                d[q][c] = s
            elif s > d[q][c]:
                d[q][c] = s
            if q not in d[c]:
                d[c][q] = s
            elif s > d[c][q]:
                d[c][q] = s
    #print(d["all_souls_000013"])
    return d


def replace_adj_weight(sim_top, database="oxford"):
    # assume that the matrix has both Q, X in rows and columns
    count = 0
    file_path = "/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/landmarks/oxfordRe/Rebuttal_NoQE_GEM.bin"
    if database == "oxford":
        qhashes, chashes = load_hashes()
        all_hashes = qhashes + chashes
        adj = np.zeros((sim_top.shape[0], sim_top.shape[0]))
        d = get_inliers(file_path)
        for i in range(sim_top.shape[0]):
            for j in range(sim_top.shape[1]):
                q = all_hashes[i]
                c = all_hashes[sim_top[i, j]]
                #print(q, c)
                if c not in d[q]:
                    adj[i,sim_top[i,j]] = 0
                    count += 1
                else:
                    adj[i,sim_top[i,j]] = d[q][c]
    #print(adj)
    print("missing " + str(count) + " pairs")
    return adj

        

def gen_graph(adj_Q_pos, Q, X, k=5, k_qe=5, do_qe=False):
    threshold = 0.7
    t = time.time()

    f = np.concatenate((Q.T, X.T))

    # sim = np.matmul(f,f.T)
    sim = np.matmul(Q.T, X)
    # sim = np.power(sim,3)
    # sim_top = np.argpartition(sim,-k,1)[:,-k:]
    sim_top = adj_Q_pos[:, 0:k]

    if do_qe:
        # Query Expansion Vector
        def qe_vec(preds, Q, X, k=2):
            Qexp = np.array([(np.sum(X[:, top[:k]], axis=1) + query) / (k + 1) for query, top in zip(Q.T, preds.T)]).T
            B = Qexp[:, 70:]
            # print("========Graph's score after DBA-QE=======")
            # eval_revop(np.argsort(-np.matmul(B.T,Qexp),axis=0))
            return np.matmul(X.T, Qexp), Qexp.T

        sim_qe, f = qe_vec(sim_top.T, f.T, f.T, k_qe)
        sim_top = np.argpartition(sim_qe, -k, 1)[:, -k:]
        # sim = sim_qe
    adj = np.zeros(sim.shape)

    for i in range(adj.shape[0]):
        adj[i, sim_top[i]] = sim[i, sim_top[i]]
        #adj[i, i] = 0
    # adj[sim_top[i], i] = sim[sim_top[i], i]
    #        for j in range(k):
    #            if adj[i,j] < threshold:
    #                adj[i,j] = 0
    #         adj[i,i]=1.0

    #    adj = adj * adj.T

    # networkx format
    # adj = np.where(adj>0, 1, 0)
    # print adj
    adj = sp.csr_matrix(adj)
    # G = nx.from_numpy_matrix(adj)
    # adj = nx.adjacency_matrix(G)
    # make symmetric for only query
    # for i in range(rows.shape[0]):
    #    if rows[i] < Q.shape[0]:
    #    adj[columns[i], rows[i]] = adj[rows[i], columns[i]] if adj[columns[i], rows[i]] == 0 else adj[columns[i], rows[i]]


    print('created G, adj with [k={}][qe={}][do_qe={}][{:.2f}s]'.format(k, k_qe, do_qe, time.time() - t))
    return adj, Q.T


def gen_graph_index(adj_pos, Q, X, k=5, k_qe=5, do_qe=False):
    threshold = 0.7
    t = time.time()

    f = X.T
    if f.shape[0] > 10000:
        # break into chunks
        chunk_size = 20000
        adj = sp.csr_matrix((f.shape[0], f.shape[0]))
        start = time.time()
        for i in range(0, f.shape[0], chunk_size):
            sim = np.matmul(f[i:i + chunk_size, :], f.T)
            sim_top = np.argpartition(sim, -k, 1)[:, -k:]
            for j in range(sim_top.shape[0]):
                adj[i + j, sim_top[j]] = sim[j, sim_top[j]]
                adj[sim_top[j], i + j] = np.expand_dims(sim[j, sim_top[j]], axis=-1)  # sim[sim_top[j], j]
                adj[i + j, i + j] = 0
            sys.stdout.write("\r" + "calculating kNN graph: [" + str(i) + "/" + str(f.shape[0]) + "] and took: " + str(
                time.time() - start))
            sys.stdout.flush()
        return adj

    # f = np.concatenate((Q.T,X.T))

    sim = np.matmul(f, f.T)
    # sim_top = np.argpartition(sim,-k,1)[:,-k:]
    sim_top = adj_pos[:, 0:k]

    if do_qe:
        # Query Expansion Vector
        def qe_vec(preds, Q, X, k=2):
            Qexp = np.array([(np.sum(X[:, top[:k]], axis=1) + query) / (k + 1) for query, top in zip(Q.T, preds.T)]).T
            B = Qexp[:, 70:]
            print("========Graph's score after DBA-QE=======")
            eval_revop(np.argsort(-np.matmul(B.T, Qexp), axis=0))
            return np.matmul(X.T, Qexp), Qexp.T

        sim_qe, f = qe_vec(sim_top.T, f.T, f.T, k_qe)
        sim_top = np.argpartition(sim_qe, -k, 1)[:, -k:]
        # sim = sim_qe

    adj = np.zeros(sim.shape)
    # sim_top = add_neighbours_neighbour(sim_top)
    for i in range(adj.shape[0]):
        adj[i, sim_top[i]] = sim[i, sim_top[i]]
        adj[sim_top[i], i] = sim[i, sim_top[i]]  # sim[sim_top[i], i]
        adj[i, i] = 0

        # for i in range(adj.shape[0]):
        #     for j in sim_top[i]:
        #         if i not in sim_top[j]:
        #             continue
        #         adj[i, j] = sim[i,j]
        #     adj[i,i] = 0
        # adj[i,sim_top[i]] = sim[i,sim_top[i]]
        # adj[sim_top[i], i] = sim[i,sim_top[i]]#sim[sim_top[i], i]
        # adj[i,i] = 0
    #        for j in range(k):
    #            if adj[i,j] < threshold:
    #                adj[i,j] = 0
    #         adj[i,i]=1.0

    #    adj = adj * adj.T

    # networkx format
    # adj = np.where(adj>0, 1, 0)
    # print adj
    # adj = replace_adj_weight(sim_top)
    # print(adj)
    G = nx.from_numpy_matrix(adj)
    adj = nx.adjacency_matrix(G)
    rows, columns = adj.nonzero()
    # make symmetric
    # for i in range(rows.shape[0]):
    #    adj[columns[i], rows[i]] = adj[rows[i], columns[i]] if adj[columns[i], rows[i]] == 0 else adj[columns[i], rows[i]]


    print('created G, adj with [k={}][qe={}][do_qe={}][{:.2f}s]'.format(k, k_qe, do_qe, time.time() - t))
    return adj, f


# def gen_graph(Q, X, k = 5, k_qe=5, do_qe=False):
#     threshold = 0.7
#     t = time.time()
#
#     f = np.concatenate((Q.T,X.T))
#
#     #sim = np.matmul(f,f.T)
#     sim = np.matmul(Q.T,X)
#     #sim = np.power(sim,3)
#     sim_top = np.argpartition(sim,-k,1)[:,-k:]
#
#     if do_qe:
#         # Query Expansion Vector
#         def qe_vec(preds,Q,X, k = 2):
#             Qexp = np.array([(np.sum(X[:,top[:k]],axis=1)+query)/(k+1) for query,top in zip(Q.T,preds.T)]).T
#             B = Qexp[:, 70:]
#             #print("========Graph's score after DBA-QE=======")
#             #eval_revop(np.argsort(-np.matmul(B.T,Qexp),axis=0))
#             return np.matmul(X.T,Qexp), Qexp.T
#
#         sim_qe, f = qe_vec(sim_top.T,f.T,f.T,k_qe)
#         sim_top = np.argpartition(sim_qe,-k,1)[:,-k:]
#         #sim = sim_qe
#     adj = np.zeros(sim.shape)
#
#     for i in range(adj.shape[0]):
#         adj[i,sim_top[i]] = sim[i,sim_top[i]]
#         #adj[i,i] = 0
# #        adj[sim_top[i], i] = sim[sim_top[i], i]
# #        for j in range(k):
# #            if adj[i,j] < threshold:
# #                adj[i,j] = 0
# #         adj[i,i]=1.0
#
# #    adj = adj * adj.T
#
#     # networkx format
#     # adj = np.where(adj>0, 1, 0)
#     # print adj
#     adj = sp.csr_matrix(adj)
#     #G = nx.from_numpy_matrix(adj)
#     #adj = nx.adjacency_matrix(G)
#     # make symmetric for only query
#     #for i in range(rows.shape[0]):
#     #    if rows[i] < Q.shape[0]:
#     #    adj[columns[i], rows[i]] = adj[rows[i], columns[i]] if adj[columns[i], rows[i]] == 0 else adj[columns[i], rows[i]]
#
#
#     print('created G, adj with [k={}][qe={}][do_qe={}][{:.2f}s]'.format(k,k_qe,do_qe,time.time()-t))
#     return adj, Q.T
#
# def gen_graph_index(Q, X, k = 5, k_qe=5, do_qe=False):
#     threshold = 0.7
#     t = time.time()
#
#     f = X.T
#     if f.shape[0] > 10000:
#         # break into chunks
#         chunk_size = 20000
#         adj = sp.csr_matrix((f.shape[0], f.shape[0]))
#         start = time.time()
#         for i in range(0, f.shape[0], chunk_size):
#             sim = np.matmul(f[i:i+chunk_size, :], f.T)
#             sim_top = np.argpartition(sim,-k,1)[:,-k:]
#             for j in range(sim_top.shape[0]):
#                 adj[i+j, sim_top[j]] = sim[j, sim_top[j]]
#                 adj[sim_top[j], i+j] = np.expand_dims(sim[j, sim_top[j]], axis=-1) #sim[sim_top[j], j]
#                 adj[i+j, i+j] = 0
#             sys.stdout.write("\r" + "calculating kNN graph: [" + str(i) + "/" + str(f.shape[0]) + "] and took: " + str(time.time() - start))
#             sys.stdout.flush()
#         return adj
#
#     #f = np.concatenate((Q.T,X.T))
#
#     sim = np.matmul(f,f.T)
#     sim_top = np.argpartition(sim,-k,1)[:,-k:]
#
#     if do_qe:
#         # Query Expansion Vector
#         def qe_vec(preds,Q,X, k = 2):
#             Qexp = np.array([(np.sum(X[:,top[:k]],axis=1)+query)/(k+1) for query,top in zip(Q.T,preds.T)]).T
#             B = Qexp[:, 70:]
#             print("========Graph's score after DBA-QE=======")
#             eval_revop(np.argsort(-np.matmul(B.T,Qexp),axis=0))
#             return np.matmul(X.T,Qexp), Qexp.T
#
#         sim_qe, f = qe_vec(sim_top.T,f.T,f.T,k_qe)
#         sim_top = np.argpartition(sim_qe,-k,1)[:,-k:]
#         #sim = sim_qe
#
#     adj = np.zeros(sim.shape)
#     #sim_top = add_neighbours_neighbour(sim_top)
#
#     for i in range(adj.shape[0]):
#         adj[i,sim_top[i]] = sim[i,sim_top[i]]
#         adj[sim_top[i], i] = sim[i,sim_top[i]]#sim[sim_top[i], i]
#         adj[i,i] = 0
# #        for j in range(k):
# #            if adj[i,j] < threshold:
# #                adj[i,j] = 0
# #         adj[i,i]=1.0
#
# #    adj = adj * adj.T
#
#     # networkx format
#     # adj = np.where(adj>0, 1, 0)
#     # print adj
#     #adj = replace_adj_weight(sim_top)
#     #print(adj)
#     G = nx.from_numpy_matrix(adj)
#     adj = nx.adjacency_matrix(G)
#     rows, columns = adj.nonzero()
#     # make symmetric
#     #for i in range(rows.shape[0]):
#     #    adj[columns[i], rows[i]] = adj[rows[i], columns[i]] if adj[columns[i], rows[i]] == 0 else adj[columns[i], rows[i]]
#
#
#     print('created G, adj with [k={}][qe={}][do_qe={}][{:.2f}s]'.format(k,k_qe,do_qe,time.time()-t))
#     return adj,f

def load_data():
    cfg,data = init_revop('roxford5k', DATA_PATH)
    Q = data['Q']
    X = data['X']
    return Q, X

def load_data_paris():
    cfg,data = init_revop('rparis6k', DATA_PATH)
    Q = data['Q']
    X = data['X']
    return Q, X

def load_hashes():
    cfg,data = init_revop('roxford5k', DATA_PATH)
    return cfg['qimlist'], cfg['imlist']

def load_hashes_paris():
    cfg,data = init_revop('rparis6k', DATA_PATH)
    return cfg['qimlist'], cfg['imlist']


def get_sim_top(i, l, d, k, start_time):
    parts = l.split(",")
    q = parts[0]
    q_pos = d[q]
    cAnds = parts[1].split(" ")
    sim_pair = [(d[cAnds[ii]], cAnds[ii+1]) for ii in range(0, k*2, 2)]
    sys.stdout.write("\r" + "finished reading [" + str(i) + "]  time taken: " + str(time.time() - start_time))
    sys.stdout.flush()
    return sim_pair

def build_adj(sim_top_line, total_length, i, start_time):
    a = sp.csr_matrix((1, total_length))
    for pos, s in sim_top_line:
        pos = int(pos)
        if float(s) > 500000:
            a[0, pos] = float(s) / 1000000
    sys.stdout.write("\r" + "writing adj:  [" + str(i) + "]  time taken: " + str(time.time() - start_time))
    sys.stdout.flush()
    return a

def load_from_prebuild(file_name, Q_features, X_features, D_features, k=5):
    Q = np.load(Q_features).T.astype(np.float32)
    X = np.load(X_features).T.astype(np.float32)
    D = np.load(D_features).T.astype(np.float32)

    # read from the prebuild file
    #f = open(file_name, "r")
    #lines = [line[:-1] for line in f.readlines()]
    #d = Manager().dict()
    #i = 0
    #for l in lines:
    #    parts = l.split(",")
    #    q = parts[0]
    #    d[q] = i
    #    i += 1
    #    sys.stdout.write("\r" + "read into hashes [" + str(i) + "/" + str(len(lines)) + "] ")
    #    sys.stdout.flush()
    #start_time = time.time()
    #sim_top = Parallel(n_jobs=120)(delayed(get_sim_top)(i, lines[i], d, k, start_time) for i in range(len(lines)))
    #sim_top = np.array(sim_top)
    ##np.save("sim_top_GEM.npy", sim_top)
    ##sim_top = np.load("sim_top_GEM.npy")
    #adj = Parallel(n_jobs=120)(delayed(build_adj)(sim_top[i], sim_top.shape[0], i, start_time) for i in range(len(sim_top)))
    #adj_stack = sp.vstack(adj)
    #adj = adj_stack
    #adj = sp.csr_matrix(adj)
    #sp.save_npz("adj_1M_GEM_knn" + str(k) + "_greater_500000.npz", adj)
    #adj = sp.load_npz("adj_1M_GEM_knn" + str(k) + "_greater_500000.npz")
    adj = sp.load_npz("/media/chundi/3b6b0f74-0ac7-42c7-b76b-00c65f5b3673/revisitop/cnnimageretrieval-pytorch/data/test/matlab_data/adj_1M_GEM_knn" + str(k) + ".npz")
    print("this should be 1 ------>" + str(adj[10000, 10000]))
    rows = range(adj.shape[0])
    adj[rows, rows] = 0
    print("Making index Symmetric")
    adj_index = adj[Q.shape[1]:, Q.shape[1]:]
    #making the adj_index to be symmetric
    rows, columns = adj_index.nonzero()
    adj_index[columns, rows] = adj_index[rows, columns]
    adj_Q = adj[:Q.shape[1], Q.shape[1]:]
    features_index = np.concatenate((X.T,D.T))
    features_Q = Q.T
    return adj_index, features_index, adj_Q, features_Q















