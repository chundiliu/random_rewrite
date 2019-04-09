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

import os

os.environ["CUDA_VISIBLE_DEVICES"]="2"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=99999, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=2048, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=512, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--batch-size', type=int, default=4096, help='Batch size.')

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

def gae_for(args, position):
    tf.set_random_seed(428)
    print("Using {} dataset".format(args.dataset_str))
    #qhashes, chashes = load_hashes()
    Q, X = load_data()
    prebuild = "/media/chundi/3b6b0f74-0ac7-42c7-b76b-00c65f5b3673/revisitop/cnnimageretrieval-pytorch/data/test/matlab_data/GEM_wDis_prebuild.bin"
    Q_features = "/media/chundi/3b6b0f74-0ac7-42c7-b76b-00c65f5b3673/revisitop/cnnimageretrieval-pytorch/data/test/matlab_data/roxford5k_GEM_lw_query_feats.npy" #"/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/landmarks/oxfordRe/evaluation/roxHD_query_fused.npy"
    X_features = "/media/chundi/3b6b0f74-0ac7-42c7-b76b-00c65f5b3673/revisitop/cnnimageretrieval-pytorch/data/test/matlab_data/roxford5k_GEM_index.npy"
    D_features = "/media/chundi/3b6b0f74-0ac7-42c7-b76b-00c65f5b3673/revisitop/cnnimageretrieval-pytorch/data/test/matlab_data/roxford5k_GEM_Dis.npy"
    #adj, features, adj_Q, features_Q = load_from_prebuild(prebuild, Q_features, X_features, D_features, k=5) # ----> 1M
    #cut_size = 800000
    #adj = adj[:cut_size, :cut_size]
    #adj_Q = adj_Q[:, :cut_size]
    #features = features[:cut_size]
    #Q = np.load("/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/landmarks/oxfordRe/evaluation/roxHD_query_fused.npy").T.astype(np.float32)
    #X = np.load("/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/landmarks/oxfordRe/evaluation/roxHD_index_fused.npy").T.astype(np.float32)
    #D = np.load("/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/landmarks/revisitop1m/revisitDistractors_fused_3s_cq.npy").T.astype(np.float32)
    #X = np.concatenate((X.T,D.T)).T
    # load the distractor too, shape should be (2048, 1M)

    adj_Q_pos = np.load('adj_q_pos_ransac_gem_complete.npy')
    adj_pos = np.load('adj_pos_ransac_gem_complete.npy')

    adj, features = gen_graph_index(adj_pos, Q, X, k=5, k_qe=3, do_qe=False)  # -----> 5k

    adj_Q, features_Q = gen_graph(adj_Q_pos, Q, X, k=10, k_qe=3,
                                  do_qe=False)

    #adj, features = gen_graph_index(Q, X, k=5, k_qe=3, do_qe=False) #-----> 5k

    #adj_Q, features_Q = gen_graph(Q, X, k=5, k_qe=3, do_qe=False) #generate validation/revop evaluation the same way as training ----> 5k
    features_all = np.concatenate([features_Q, features])

    #adj_Q = adj_Q.todense()
    #adj_all = np.concatenate([adj_Q, adj.todense()])
    #adj_all = np.pad(adj_all, [[0,0], [Q.shape[1], 0]], "constant")
    adj_all = sp.vstack((adj_Q, adj))
    zeros = sp.csr_matrix((adj_all.shape[0], Q.shape[1]))
    adj_all = sp.hstack((zeros, adj_all))
    adj_all = sp.csr_matrix(adj_all)
    rows, columns = adj_all.nonzero()
    print("preprocessing adj_all")
    adj_all_norm = preprocess_graph(adj_all)
    #adj = add_neighbours_neighbour(adj)
    #adj1, features1 = load_data(args.dataset_str)
    #features = torch.from_numpy(features)
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
    #features_evaluate = torch.from_numpy(features_evaluate)
    # validation done\\\

    features_pre = (adj_norm * features)
    #features_pre = features_pre / np.linalg.norm(features_pre, ord=2, axis=1).reshape((features_pre.shape[0], 1))
    features_all_pre = (adj_all_norm * features_all)
    #features_all_pre = features_all_pre / np.linalg.norm(features_all_pre, ord=2, axis=1).reshape((features_all_pre.shape[0], 1))
    revop_map = get_roc_score_matrix(features_all_pre, Q.shape[1])
    print('inference: {}'.format(revop_map))
    #ipdb.set_trace()

    #with tf.device('/device:GPU:1'):
    #adj_label_spt = convert_sparse_matrix_to_sparse_tensor(adj_label)
    featts_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2048])



    #ipdb.set_trace()
    regularizer = tf.contrib.layers.l2_regularizer(scale=1e-5)

    print(features_pre.shape[0])
    training_dataset = tf.data.Dataset.from_tensor_slices(featts_ph)
    training_dataset = training_dataset.shuffle(buffer_size=10000)
    training_dataset = training_dataset.batch(args.batch_size, drop_remainder=True)
    training_dataset = training_dataset.repeat()

    validation_dataset = tf.data.Dataset.from_tensor_slices(featts_ph)
    validation_dataset = validation_dataset.batch(2600)

    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                               training_dataset.output_shapes)
    itr_train_init_op = iterator.make_initializer(training_dataset)
    #ipdb.set_trace()
    itr_valid_init_op = iterator.make_initializer(validation_dataset)
    next_element = iterator.get_next()

    #ipdb.set_trace()

    # 1st GCN
    with tf.variable_scope('GCN1'):
        W1 = tf.get_variable(name="w", shape=[feat_dim,args.hidden1], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(),
                             regularizer=regularizer)
        B1 = tf.get_variable(name='b', shape=[args.hidden1], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))
        output1 = tf.nn.bias_add(tf.matmul(next_element, W1), B1)
        output1 = tf.nn.elu(output1)

    with tf.variable_scope('InnerProduct'):
        hidden_emb = tf.nn.l2_normalize(output1, axis=1)
        #hidden_emb = output1
        adj_preds = tf.matmul(hidden_emb, tf.transpose(hidden_emb))
        adj_preds = tf.nn.relu(adj_preds)
        #adj_preds = tf.nn.dropout(adj_preds, 0.99)

    #ipdb.set_trace()
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # losses = tf.nn.weighted_cross_entropy_with_logits(
    #     #tf.zeros_like(adj_preds, dtype=tf.float32),
    #     adj_preds,
    #     adj_preds,
    #     pos_weight=1.0,
    #     name='weighted_loss'
    # )

    #losses = tf.keras.backend.binary_crossentropy(adj_preds, adj_preds, False)

    losses = -1 * adj_preds ** 2 + 0.5 * adj_preds + 0.5

    global_step = tf.Variable(0, trainable=False)
    loss = tf.reduce_mean(losses)
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    loss += reg_term
    learning_rate_t = tf.train.exponential_decay(args.lr, global_step, 9999999999, 0.3)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_t, beta1=0.0, beta2=0.99999, epsilon=1e-12)
    #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_t, momentum=0.0)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)
    train_init_op = tf.global_variables_initializer()

    best_revop = 0.0

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=session_conf) as sess:
        #sess.run(itr_train_init_op, feed_dict={featts_ph: features})
        #sess.run(iterator2.initializer)

        sess.run(train_init_op)
        #ipdb.set_trace()
        #hidden_emb_inf = sess.run(step2_norm)
        #revop_map = get_roc_score_matrix(hidden_emb_inf, Q.shape[1])
        #print("inference map:{}".format(revop_map))
        #sess.run(train_init_op)

        itr = 0
        sess.run(itr_train_init_op, feed_dict={featts_ph: features_pre})
        while itr < args.epochs:
            start_time = time.time()
            sess.run(global_step.assign(itr + 1))
            #sess.run(tf.initialize_variables([labels]))
            _, loss_out = sess.run([train_op, loss])
            end_time = time.time() - start_time
            print(
                "itr: {}, train loss: {}, train time: {}".format(itr, loss_out, str(end_time)))
            itr += 1
            if loss_out <= 0:
                ipdb.set_trace()


            if itr % 1 == 0:
                start_time = time.time()
                sess.run(itr_valid_init_op, feed_dict={featts_ph: features_all_pre})
                first_time_flag = True
                hidden_emb_np = None
                while True:
                    try:
                        if first_time_flag:
                            hidden_emb_np = sess.run(hidden_emb)
                            first_time_flag = False
                        else:
                            hidden_emb_np = np.concatenate([hidden_emb_np, sess.run(hidden_emb)], axis=0)
                    except tf.errors.OutOfRangeError:
                        break
                end_time = time.time() - start_time
                revop_map = get_roc_score_matrix(hidden_emb_np, Q.shape[1])
                if revop_map > best_revop:
                    best_revop = revop_map
                print("train loss: {}, inf time: {}, revop:{}, best revop:{}".format(loss_out, str(end_time),revop_map, best_revop))
                sess.run(itr_train_init_op, feed_dict={featts_ph: features_pre})
    


if __name__ == '__main__':
    gae_for(args, 0)

