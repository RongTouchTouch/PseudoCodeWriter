import torch
import torch.nn as nn
import math

from models.attention import MultiHeadAttention


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Block(nn.Module):
    def __init__(self,
                 num_units,
                 query_dim,
                 key_dim,
                 dropout=0.5,
                 h=8):
        super(Block, self).__init__()
        self.layer_norm1 = nn.LayerNorm(normalized_shape=num_units) # CxHxW
        self.layer_norm2 = nn.LayerNorm(normalized_shape=num_units)
        self.linear_layer1 = nn.Linear(in_features=num_units, out_features=num_units*h)
        self.linear_layer2 = nn.Linear(in_features=num_units, out_features=num_units*h)
        # self.dropout = dropout
        self.dropout = nn.Dropout(dropout)
        self.attention = MultiHeadAttention(query_dim, key_dim, num_units, dropout, h)
        self.activation = nn.PReLU(num_units*h)
        self.graph_activation = nn.PReLU(num_units*h)

    def forward(self, query, keys):
        out = self.attention(query, keys)
        out = self.layer_norm1(out)
        out = self.dropout(self.linear_layer1(self.linear_layer2(out) + self.layer_norm2(out)))
        out = self.linear_layer1(out + keys)
        return out


class GraphEncoder:
    def __init__(self,
                 num_embeddings,  # size of relations?
                 num_units,
                 encoder_size=6,
                 sparse=True,):
        super(GraphEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=num_units)
        nn.init.xavier_normal_(self.embedding.weight)

        self.sparse = sparse
        self.encoder_size = encoder_size
        self.gat = nn.ModuleList([Block(num_units, num_units, num_units) for _ in range(encoder_size)])

    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])

    def forward(self, adj_matrix, relation, entity, entity_length):
        if 0:
            processed_entity = torch.tensor(entity, requires_grad=False)
        else:
            processed_entity = entity
        processed_relation = [self.embedding(x) for x in relation]
        glob = [] # global vector
        graphs = []
        for i, adj in enumerate(adj_matrix):
            vgraph = torch.cat((processed_entity[i][:entity_length[i]], processed_relation[i]), 0)
            N = vgraph.size(0)
            if self.sparse:
                lens = [len(x) for x in adj]
                m = max(lens)
                mask = torch.arange(0, m).unsqueeze(0).repeat(len(lens), 1).long()
                mask = (mask <= torch.LongTensor(lens).unsqueeze(1)).cuda()
                mask = (mask == 0).unsqueeze(1)
            else:
                mask = (adj == 0).unsqueeze(1)
            for j in range(self.encoder_size):
                if self.sparse:
                    ngraph = [vgraph[k] for k in adj]
                    ngraph = [self.pad(x, m) for x in ngraph]
                    ngraph = torch.stack(ngraph, 0)
                    # print(ngraph.size(),vgraph.size(),mask.size())
                    vgraph = self.gat[j](vgraph.unsqueeze(1), ngraph, mask)
                else:
                    ngraph = torch.tensor(vgraph.repeat(N, 1).view(N, N, -1), requires_grad=False)
                    vgraph = self.gat[j](vgraph.unsqueeze(1), ngraph, mask)
                    if self.args.model == 'gat':
                        vgraph = vgraph.squeeze(1)
                        vgraph = self.gatact(vgraph)
            graphs.append(vgraph)
            glob.append(vgraph[entity_length[i]])
        elens = [x.size(0) for x in graphs]
        gents = [self.pad(x, max(elens)) for x in graphs]
        gents = torch.stack(gents, 0)
        elens = torch.LongTensor(entity_length)
        emask = torch.arange(0, gents.size(1)).unsqueeze(0).repeat(gents.size(0), 1).long()
        emask = (emask <= elens.unsqueeze(1)).cuda()
        glob = torch.stack(glob, 0)
        return None, glob, (gents, emask)

