from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import sys

from gae.utils import mask_test_edges, preprocess_graph_sp, preprocess_graph, get_roc_score, get_roc_score_matrix, \
    mask_test_rows, sparse_mx_to_torch_sparse_tensor, neg_sample
from gae.preprocess_graph import *
from joblib import Parallel, delayed
import ipdb
import ctypes
from sklearn.manifold import TSNE

import tensorflow as tf

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=99999, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=2048, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=512, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--batch-size', type=int, default=6323, help='Batch size.')

# AE, VAE, AE_batch, VAE_batch
mode = "AE"
print("mode is " + mode)
args = parser.parse_args()
for key in vars(args):
    print(key + ":" + str(vars(args)[key]))


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensorValue(indices, coo.data.astype(np.float32), coo.shape)


def gae_for(args, aa):
    tf.set_random_seed(428)
    tf.reset_default_graph()
    print("Using {} dataset".format(args.dataset_str))
    # qhashes, chashes = load_hashes()
    Q, X = load_data_paris()
    prebuild = "/media/chundi/3b6b0f74-0ac7-42c7-b76b-00c65f5b3673/revisitop/cnnimageretrieval-pytorch/data/test/matlab_data/GEM_wDis_prebuild.bin"
    Q_features = "/media/chundi/3b6b0f74-0ac7-42c7-b76b-00c65f5b3673/revisitop/cnnimageretrieval-pytorch/data/test/matlab_data/roxford5k_GEM_lw_query_feats.npy"  # "/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/landmarks/oxfordRe/evaluation/roxHD_query_fused.npy"
    X_features = "/media/chundi/3b6b0f74-0ac7-42c7-b76b-00c65f5b3673/revisitop/cnnimageretrieval-pytorch/data/test/matlab_data/roxford5k_GEM_index.npy"
    D_features = "/media/chundi/3b6b0f74-0ac7-42c7-b76b-00c65f5b3673/revisitop/cnnimageretrieval-pytorch/data/test/matlab_data/roxford5k_GEM_Dis.npy"
    # adj, features, adj_Q, features_Q = load_from_prebuild(prebuild, Q_features, X_features, D_features, k=5) # ----> 1M
    # cut_size = 800000
    # adj = adj[:cut_size, :cut_size]
    # adj_Q = adj_Q[:, :cut_size]
    # features = features[:cut_size]
    # Q = np.load("/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/landmarks/oxfordRe/evaluation/roxHD_query_fused.npy").T.astype(np.float32)
    # X = np.load("/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/landmarks/oxfordRe/evaluation/roxHD_index_fused.npy").T.astype(np.float32)
    # D = np.load("/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/landmarks/revisitop1m/revisitDistractors_fused_3s_cq.npy").T.astype(np.float32)
    # X = np.concatenate((X.T,D.T)).T
    # load the distractor too, shape should be (2048, 1M)

    # adj_Q_pos = np.load('adj_q_pos_ransac_gem_paris.npy')
    # adj_pos = np.load('adj_pos_ransac_gem_paris.npy')

    adj, features = gen_graph_index(Q, X, k=5, k_qe=3, do_qe=False)  # -----> 5k

    adj_Q, features_Q = gen_graph(Q, X, k=15, k_qe=3, do_qe=False)

    # adj, features = gen_graph_index(Q, X, k=5, k_qe=3, do_qe=False) #-----> 5k

    # adj_Q, features_Q = gen_graph(Q, X, k=5, k_qe=3, do_qe=False) #generate validation/revop evaluation the same way as training ----> 5k
    features_all = np.concatenate([features_Q, features])

    # adj_Q = adj_Q.todense()
    # adj_all = np.concatenate([adj_Q, adj.todense()])
    # adj_all = np.pad(adj_all, [[0,0], [Q.shape[1], 0]], "constant")
    adj_all = sp.vstack((adj_Q, adj))
    zeros = sp.csr_matrix((adj_all.shape[0], Q.shape[1]))
    adj_all = sp.hstack((zeros, adj_all))
    adj_all = sp.csr_matrix(adj_all)
    rows, columns = adj_all.nonzero()
    print("preprocessing adj_all")
    adj_all_norm = preprocess_graph(adj_all)
    # adj = add_neighbours_neighbour(adj)
    # adj1, features1 = load_data(args.dataset_str)
    # features = torch.from_numpy(features)
    # features_all = torch.from_numpy(features_all)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj = adj_orig

    # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    print("Sampling validation")
    # adj_train, adj_val, features, features_valid = mask_test_rows(adj, features)
    adj_train = adj
    adj = adj_train

    # Some preprocessing
    print("preprocessing adj")
    adj_norm = preprocess_graph(adj)
    # adj_norm_label = preprocess_graph_sp(adj)

    adj_label = adj_train + sp.eye(
        adj_train.shape[0])  # adj_norm_label + sp.eye(adj_train.shape[0]) #adj_train + sp.eye(adj_train.shape[0])
    # rows, columns = adj_label.nonzero()
    # adj_label[columns, rows] = adj_label[rows, columns]
    # adj_label = sparse_to_tuple(adj_label)
    # adj_label = torch.FloatTensor(adj_label.toarray())

    print("adj sum: " + str(adj.sum()))
    pos_weight = float(float(adj.shape[0]) * adj.shape[0] - adj.sum()) / adj.sum()
    print("top part: " + str(float(float(adj.shape[0]) * adj.shape[0] - adj.sum())))
    print("pos wieght: " + str(pos_weight))
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    features_pre = (features)
    # features_pre = features_pre / np.linalg.norm(features_pre, ord=2, axis=1).reshape((features_pre.shape[0], 1))
    features_all_pre = (features_all)
    # features_all_pre = features_all_pre / np.linalg.norm(features_all_pre, ord=2, axis=1).reshape((features_all_pre.shape[0], 1))
    revop_map = get_roc_score_matrix(features_all_pre, Q.shape[1])
    print('inference: {}'.format(revop_map))

    # ipdb.set_trace()

    def run_tf(EARLY_STOPPING_ITR=50, alpha=2, beta=0.3, init1=1e-5, init2=1e-5, reg=5e-4, learning_rate=0.0001):
        tf.reset_default_graph()
        # with tf.device('/device:GPU:1'):
        # adj_label_spt = convert_sparse_matrix_to_sparse_tensor(adj_label)
        featts_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2048])

        adj_norm_spt = convert_sparse_matrix_to_sparse_tensor(adj_norm)
        adj_all_norm_spt = convert_sparse_matrix_to_sparse_tensor(adj_all_norm)
        adj_ph = tf.sparse_placeholder(dtype=tf.float32, shape=[None, None])

        # ipdb.set_trace()
        regularizer = tf.contrib.layers.l2_regularizer(scale=reg)

        # print(features_pre.shape[0])
        training_dataset = tf.data.Dataset.from_tensor_slices(featts_ph)
        # training_dataset = training_dataset.shuffle(buffer_size=10000)
        training_dataset = training_dataset.batch(args.batch_size)
        training_dataset = training_dataset.repeat()

        validation_dataset = tf.data.Dataset.from_tensor_slices(featts_ph)
        validation_dataset = validation_dataset.batch(6392)

        iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                                   training_dataset.output_shapes)
        itr_train_init_op = iterator.make_initializer(training_dataset)
        # ipdb.set_trace()
        itr_valid_init_op = iterator.make_initializer(validation_dataset)
        next_element = iterator.get_next()

        # ipdb.set_trace()

        # # 1st GCN
        with tf.variable_scope('GCN1'):
            # np.random.seed(428)
            init_w = (np.random.randn(args.hidden1, args.hidden1) * init1)
            # init_w = np.zeros((args.hidden1,args.hidden1))
            init_w[np.where(np.eye(args.hidden1) != 0)] = 1
            # ipdb.set_trace()
            # print(init_w)
            constant_init = tf.convert_to_tensor(init_w, dtype=tf.float32)
            W1 = tf.get_variable(name="w", dtype=tf.float32,
                                 initializer=constant_init,
                                 regularizer=regularizer)
            B1 = tf.get_variable(name='b', shape=[args.hidden1], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))
            walk1 = tf.sparse_tensor_dense_matmul(adj_ph, next_element)
            output1 = tf.nn.bias_add(tf.matmul(walk1, W1), B1)
            output1 = tf.nn.elu(output1)
            # output1 = tf.nn.l2_normalize(output1, axis=1)
        with tf.variable_scope('GCN2'):
            # np.random.seed(428)
            init_w = (np.random.randn(args.hidden1, args.hidden1) * init2)
            # init_w = np.zeros((args.hidden1,args.hidden1))
            init_w[np.where(np.eye(args.hidden1) != 0)] = 1
            # ipdb.set_trace()
            # print(init_w)0
            constant_init = tf.convert_to_tensor(init_w, dtype=tf.float32)
            W2 = tf.get_variable(name="w", dtype=tf.float32,
                                 initializer=constant_init,
                                 regularizer=regularizer)
            B2 = tf.get_variable(name='b', shape=[args.hidden1], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))
            walk2 = tf.sparse_tensor_dense_matmul(adj_ph, output1)
            output2 = tf.nn.bias_add(tf.matmul(walk2, W2), B2)
            output2 = tf.nn.elu(output2) + 5 * output1
            # output2 = tf.nn.l2_normalize(output2, axis=1)
            # output2 = tf.sparse_tensor_dense_matmul(adj_ph, output2)

        # 1st GCN
        # with tf.variable_scope('GCN1'):
        #     W1 = tf.get_variable(name="w", shape=[feat_dim,args.hidden1], dtype=tf.float32,
        #                          initializer=tf.random_normal_initializer(),
        #                          regularizer=regularizer)
        #     B1 = tf.get_variable(name='b', shape=[args.hidden1], dtype=tf.float32,
        #                              initializer=tf.constant_initializer(0.0))
        #     output1 = tf.nn.bias_add(tf.matmul(next_element, W1), B1)
        #     output1 = tf.nn.elu(output1)

        with tf.variable_scope('InnerProduct'):
            hidden_emb = tf.nn.l2_normalize(output2, axis=1)
            # hidden_emb = output1
            adj_preds = tf.matmul(hidden_emb, tf.transpose(hidden_emb))
            adj_preds = tf.clip_by_value(adj_preds, 0.0001, 0.9999)
            # adj_preds = tf.nn.dropout(adj_preds, 0.99)

        # losses = tf.nn.weighted_cross_entropy_with_logits(
        #     #tf.zeros_like(adj_preds, dtype=tf.float32),
        #     adj_preds,
        #     adj_preds,
        #     pos_weight=1.0,
        #     name='weighted_loss'
        # )

        # losses = tf.keras.backend.binary_crossentropy(adj_preds, adj_preds, False)

        # losses = -1 * adj_preds ** 2 + 0.5 * adj_preds + 0.5

        losses = -0.5 * alpha * (adj_preds - beta) ** 2

        global_step = tf.Variable(0, trainable=False)
        loss = tf.reduce_mean(losses)
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
        loss += reg_term
        learning_rate_t = tf.train.exponential_decay(args.lr, global_step, 9999999999, 0.3)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_t, beta1=0.0, beta2=0.99999, epsilon=1e-12)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_t, momentum=0.0)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars)
        train_init_op = tf.global_variables_initializer()

        best_revop = 0.0
        stopping_tracker = 0
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=session_conf) as sess:
            # sess.run(itr_train_init_op, feed_dict={featts_ph: features})
            # sess.run(iterator2.initializer)

            sess.run(train_init_op)
            # ipdb.set_trace()
            # hidden_emb_inf = sess.run(step2_norm)
            # revop_map = get_roc_score_matrix(hidden_emb_inf, Q.shape[1])
            # print("inference map:{}".format(revop_map))
            # sess.run(train_init_op)

            itr = 0
            sess.run(itr_train_init_op, feed_dict={featts_ph: features_pre})
            while itr < args.epochs:
                start_time = time.time()
                # sess.run(global_step.assign(itr + 1))
                # sess.run(tf.initialize_variables([labels]))
                # ipdb.set_trace()
                _, loss_out = sess.run([train_op, loss], feed_dict={adj_ph: adj_norm_spt})
                end_time = time.time() - start_time
                # print(
                #     "itr: {}, train loss: {}, train time: {}".format(itr, loss_out, str(end_time)))
                itr += 1

                if itr % 1 == 0:
                    start_time = time.time()
                    sess.run(itr_valid_init_op, feed_dict={featts_ph: features_all_pre})
                    first_time_flag = True
                    hidden_emb_np = None
                    while True:
                        try:
                            if first_time_flag:
                                hidden_emb_np = sess.run(hidden_emb, feed_dict={adj_ph: adj_all_norm_spt})
                                first_time_flag = False
                            else:
                                break
                                hidden_emb_np = np.concatenate([hidden_emb_np, sess.run(hidden_emb)], axis=0)
                        except tf.errors.OutOfRangeError:
                            break
                    end_time = time.time() - start_time
                    revop_map = get_roc_score_matrix(hidden_emb_np, Q.shape[1])
                    if revop_map > best_revop:
                        best_revop = revop_map
                        stopping_tracker = 0
                    else:
                        stopping_tracker += 1
                        if stopping_tracker >= EARLY_STOPPING_ITR:
                            break
                    print("train loss: {}, inf time: {}, revop:{}, best revop:{}".format(loss_out, str(end_time),
                                                                                         revop_map, best_revop))
                    sess.run(itr_train_init_op, feed_dict={featts_ph: features_pre})
            return best_revop

    best_map = 0
    best_alpha = 0
    best_beta = 0
    best_init1 = 0
    best_init2 = 0
    best_reg = 0
    best_learning_rate = 0
    for a in [1]:
        for b in [0.25]:
            for init1 in [1e-4]:
                for init2 in [1e-12]:
                    for reg in [1e-5]:
                        for learning_rate in [0.0003]:
                            # ap = run_tf(alpha=a, beta=b)
                            ap = run_tf(25, a, b, init1, init2, reg, learning_rate)
                            # ipdb.set_trace()
                            print("{:.7f} {:.7f} {:.7f} {:.7f} {:.7f} {:.7f} {:.7f}".format(a, b, init1,
                                                                                            init2, reg, learning_rate,
                                                                                            ap))
                            if ap > best_map:
                                best_map = ap
                                best_alpha = a
                                best_beta = b
                                best_init1 = init1
                                best_init2 = init2
                                best_reg = reg
                                best_learning_rate = learning_rate
    print("\n\nbest: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(best_alpha, best_beta, best_init1,
                                                                              best_init2, best_reg, best_learning_rate,
                                                                              best_map))


if __name__ == '__main__':
    gae_for(args, 0)

