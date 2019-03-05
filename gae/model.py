import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time

from gae.layers import GraphConvolution
import numpy as np

class GCNModelAE_batch(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelAE_batch, self).__init__()
        #self.gc1 = ChebConv(input_feat_dim, hidden_dim1, 3) #GCNConv(input_feat_dim, hidden_dim1, improved=False) #GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        #self.gc2 = ChebConv(hidden_dim1, hidden_dim2, 3) #GCNConv(hidden_dim1, hidden_dim2, improved=False) #GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        #self.gc3 = ChebConv(hidden_dim1, hidden_dim2, 3) #GCNConv(hidden_dim1, hidden_dim2, improved=False) #GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.elu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        #self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.dc = InnerProductDecoder_batch(dropout, act=F.relu)

    def encode(self, x, adj, edge_weight=None):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj)

    def forward(self, x, adj, labels, n_nodes, norm, pos_weight, opt, edge_weight=None, just_mu=False, training=False):
        mu = self.encode(x, adj, edge_weight=edge_weight)
        mu = F.normalize(mu, p=2, dim=1)
        z = mu
        if just_mu:
            return mu
        adj, cost = self.dc(z, mu, 0, labels, n_nodes, norm, pos_weight, opt, training)
        return adj, mu, cost

class GCNModelVAE_batch(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE_batch, self).__init__()
        #self.gc1 = ChebConv(input_feat_dim, hidden_dim1, 3) #GCNConv(input_feat_dim, hidden_dim1, improved=False) #GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        #self.gc2 = ChebConv(hidden_dim1, hidden_dim2, 3) #GCNConv(hidden_dim1, hidden_dim2, improved=False) #GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        #self.gc3 = ChebConv(hidden_dim1, hidden_dim2, 3) #GCNConv(hidden_dim1, hidden_dim2, improved=False) #GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=lambda x: x)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        #self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.dc = InnerProductDecoder_batch(dropout, act=F.relu)

    def encode(self, x, adj, edge_weight=None):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj, labels, n_nodes, norm, pos_weight, opt, edge_weight=None, just_mu=False, training=False):
        mu, logvar = self.encode(x, adj, edge_weight=edge_weight)
        mu = F.normalize(mu, p=2, dim=1)
        z = self.reparameterize(mu, logvar)
        if just_mu:
            return mu
        adj, cost = self.dc(z, mu, logvar, labels, n_nodes, norm, pos_weight, opt, training)
        return adj, mu, logvar, cost


class InnerProductDecoder_batch(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder_batch, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z, mu, logvar, labels, n_nodes, norm, pos_weight, opt, training):
        z = F.dropout(z, self.dropout, training=self.training)
        # do this in batches?
        batch_size = 32
        num_updates = 1000
        update = 0
        total_cost = 0
        rand = np.random.permutation(z.shape[0])
        rand2 = np.random.permutation(z.shape[0])
        rows, cols = np.nonzero(labels)
        for i in range(0, z.shape[0], batch_size):
            # get the tiem to do forward passing
            batch = rand[i:i+batch_size] #range(i, min(i+batch_size, z.shape[0])) #rand[i:i+batch_size]
            batch2 = rand2[i:i+batch_size]
            forward_time = time.time()
            adj = self.act(torch.mm(z[batch], z[batch2].t()))
            #adj = z[batch]*z[batch2]
            #adj = adj.sum(dim=1)
            #adj = adj.unsqueeze(0)
            forward_time_elapse = time.time() - forward_time

            preds = adj
            label_batch = torch.FloatTensor(labels[batch, :][:, batch2].toarray())
            cost = norm * F.binary_cross_entropy_with_logits(preds, label_batch, pos_weight=pos_weight)
            #if training:
            #    backprob_time = time.time()
            #    cost.backward(retain_graph=True)
            #    backprob_time_elapse = time.time() - backprob_time
            #    opt_time = time.time()
            #    opt.step()
            #    opt_time_elapse = time.time() - opt_time
            #    sys.stdout.write("\r" + "calculating batch loss of [" + str(i) + "/" + str(z.shape[0]) + "]  and current cost: "  + str(cost) + "  forward time: " + str(forward_time_elapse) + \
            #            "  backprob time: " + str(backprob_time_elapse) + "  opt time: " + str(opt_time_elapse))
            #    sys.stdout.flush()
            sys.stdout.write("\r" + "calculating batch loss of [" + str(i) + "/" + str(z.shape[0]) + "]  and current cost: "  + str(cost))
            sys.stdout.flush()
            total_cost += cost
            update += 1
            if update == num_updates:
                break
        sys.stdout.write("\r")
        #KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        #    1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        cost = total_cost
        #cost += KLD
        if training:
            backtime = time.time()
            cost.backward()
            backtime_done = time.time() - backtime
            opt.step()
            sys.stdout.write("\r time taken to do backprop: " + str(backtime_done) + "   opt: " + str(time.time() - backtime) + "\n")
            sys.stdout.flush()
        return adj, cost


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        #self.gc1 = ChebConv(input_feat_dim, hidden_dim1, 3) #GCNConv(input_feat_dim, hidden_dim1, improved=False) #GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        #self.gc2 = ChebConv(hidden_dim1, hidden_dim2, 3) #GCNConv(hidden_dim1, hidden_dim2, improved=False) #GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        #self.gc3 = ChebConv(hidden_dim1, hidden_dim2, 3) #GCNConv(hidden_dim1, hidden_dim2, improved=False) #GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        #self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=F.relu)

    def encode(self, x, adj, edge_weight=None):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj, edge_weight=None):
        mu, logvar = self.encode(x, adj, edge_weight=edge_weight)
        mu = F.normalize(mu, p=2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


class GCNModelAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelAE, self).__init__()
        #self.gc1 = ChebConv(input_feat_dim, hidden_dim1, 3) #GCNConv(input_feat_dim, hidden_dim1, improved=False) #GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        #self.gc2 = ChebConv(hidden_dim1, hidden_dim2, 3) #GCNConv(hidden_dim1, hidden_dim2, improved=False) #GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        #self.gc3 = ChebConv(hidden_dim1, hidden_dim2, 3) #GCNConv(hidden_dim1, hidden_dim2, improved=False) #GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.elu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.elu)
        #self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        #self.dc = InnerProductDecoder(dropout, act=F.relu)

    def encode(self, x, adj, edge_weight=None):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj)

    def forward(self, x, adj, edge_weight=None):
        mu = self.encode(x, adj, edge_weight=edge_weight)
        z = F.normalize(mu, p=2, dim=1)
        return z



class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        #adj = self.act(torch.mm(z, z.t()))
        return z
