import torch
import torch.nn as nn
from models.attention import MultiHeadAttention, MatrixAttn
from models.list_encoder import list_encode, lseq_encode
from models.graph_attention import GraphEncoder
from models.beam import Beam
from models.splan import splanner
from models.graph_attention import GraphEncoder

class Model(nn.Module):
    def __init__(self,
                 num_units,
                 query_dim,
                 key_dim,
                 num_embeddings,
                 dropout=0.5,
                 h=8):
        super(Model, self).__init__()
        self.attention = MultiHeadAttention(query_dim=query_dim, key_dim=key_dim, num_units=num_units,
                                            dropout=dropout, h=h)
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=num_units)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=num_units)  # CxHxW
        self.layer_norm2 = nn.LayerNorm(normalized_shape=num_units)
        self.linear_layer1 = nn.Linear(in_features=num_units, out_features=num_units * h)
        self.linear_layer2 = nn.Linear(in_features=num_units, out_features=num_units * h)
        self.lstm = nn.LSTMCell(input_size=num_units * 3, hidden_size=num_units) # cattimes?
        self.matrix_attention = MatrixAttn()

        self.graph_encoder = GraphEncoder(num_embeddings=num_embeddings, num_units=num_units)
        self.

    def forward(self, *input):


