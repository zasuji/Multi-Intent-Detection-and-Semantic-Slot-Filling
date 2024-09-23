import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from models.hyperbolic import *


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.concat = concat
        self.alpha = alpha

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a, self.alpha)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h, g):
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        if self.concat:
            return F.elu(g.ndata.pop('h'))
        else:
            return g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, nclass, dropout, alpha, num_heads, n_layers=2):
        super(MultiHeadGATLayer, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList()
        self.nhead = num_heads
        for i in range(num_heads):
            self.attentions.append(GATLayer(in_dim, out_dim, alpha, concat=True))
        self.n_layers = n_layers
        if self.n_layers > 2:
            self.layers = nn.ModuleList()
            for i in range(self.n_layers-2):
                for j in range(num_heads):
                    self.layers.append(GATLayer(out_dim*num_heads, out_dim, alpha, concat=True))
        self.out_att = GATLayer(out_dim*num_heads, nclass, alpha, concat=False)

    def forward(self, h, g):
        batch_size = h.shape[0]
        seq_len_max = h.shape[1]
        num_dim = h.shape[2]
        h = h.reshape(batch_size*seq_len_max, num_dim)
        h = F.dropout(h, self.dropout, training=self.training)
        input = h
        h = torch.cat([attn_head(h, g) for attn_head in self.attentions], dim=1)
        if self.n_layers > 2:
            for i in range(self.n_layers - 2):
                temp = []
                h = F.dropout(h, self.dropout, training=self.training)
                cur_input =h
                for j in range(self.nhead):
                    temp.append(self.layers[i*self.nhead+j](h, g))
                h = torch.cat(temp, dim=1) + cur_input
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.elu(self.out_att(h, g))
        return (h+input).view(batch_size, seq_len_max, num_dim)
    

class heterogeneousGNN(nn.Module):
    def __init__(self, r_dim, g):
        super(heterogeneousGNN, self).__init__()
        self.r_dim = r_dim
        self.n_relations = max(g.edata['type'])+1
        self.relation_embed = nn.Parameter(torch.FloatTensor(self.n_relations, self.r_dim))  
        nn.init.normal_(self.relation_embed.data)
    
    def forward(self, int_embed, slot_embed, g):
        g = g.local_var()

        g_inv = dgl.graph((g.edges()[1], g.edges()[0]))
        embed = torch.cat([slot_embed, int_embed], dim=0)
        relation_emb = self.relation_embed
        g.ndata['node'] = embed
        g.ndata['node1'] = expmap0(embed)

        def tan_sum(edges):
            tan_sum = logmap0(expmap(ptransp0(edges.src['node1'], edges.dst['node'] + relation_emb[edges.data['type']]), edges.src['node1']))
            return {'tan_sum': tan_sum}
        
        g.apply_edges(tan_sum, g.edges(form='all')[2])
        g_inv.edata['tan_sum'] = g.edata['tan_sum']
        g_inv.update_all(dgl.function.copy_e('tan_sum','temp'),dgl.function.mean('temp','out'))

        slot_int_embed = g_inv.ndata.pop('out')

        return slot_int_embed[:slot_embed.shape[0]], slot_int_embed[slot_embed.shape[0]:]
    
    def graph_kg_loss(self, int_embed, slot_embed, g):
        g = g.local_var()

        g_inv = dgl.graph((g.edges()[1], g.edges()[0]))

        relation_emb = self.relation_embed
        embed = torch.cat([slot_embed, int_embed], dim=0)
        g.ndata['node'] = embed
        g.ndata['node1'] = expmap0(embed)

        def kg_loss(edges):
            sub = hyp_distance(expmap(ptransp0(edges.src['node1'], edges.dst['node'] + relation_emb[edges.data['type']]), edges.src['node1']), edges.src['node1'])
            return {'sub': sub}
        
        g.apply_edges(kg_loss, g.edges(form='all')[2])
        g_inv.edata['sub'] = g.edata['sub']
        g_inv.update_all(dgl.function.copy_e('sub','temp'),dgl.function.mean('temp','out'))
        
        embed_sub = g_inv.ndata.pop('out')
        loss = torch.sum(embed_sub)/(torch.sum(embed_sub>0))
        return loss
