import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class BahdanauAttention(nn.Module):
    """
    (Simplified) Bahdanau Attention (https://arxiv.org/abs/1409.0473)
    Implementation is very similar to pytorch.seq2seq.models.attention.BahdanauAttention
    """
    def __init__(self, num_units, query_size, memory_size):
        """

        :param num_units: internal feature dimension
        :param query_size: feature dimension for query
        :param memory_size: feature dimension for memory (value)
        :param batch_first: if True batch size is the 1st dimension, if False
            the sequence is first and batch size is second
        """
        super(BahdanauAttention, self).__init__()
        self._num_units = num_units
        self._softmax = nn.Softmax(dim=-1)
        self._tanh = nn.Tanh()

        self.query_layer = nn.Linear(query_size, num_units, bias=False)
        self.memory_layer = nn.Linear(memory_size, num_units, bias=False)
        self.alignment_layer = nn.Linear(num_units, 1, bias=False)
        self.linear_att = Parameter(torch.Tensor(num_units))

    def calc_score(self, att_query, att_keys):
        """
        Calculate Bahdanau score

        :param att_query: b x t_q x n
        :param att_keys: b x t_k x n
        :return: b x t_q x t_k scores
        """

        b, t_k, n = att_keys.size()
        t_q = att_query.size(1)

        att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n)
        att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n)
        sum_qk = att_query + att_keys

        linear_att = self.linear_att

        out = torch.tanh(sum_qk).matmul(linear_att)

        return out

    def forward(self, query, keys):
        """

        :param query: class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
        :param keys: class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                over which to apply the attention mechanism.

        :return:
              context: class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
              weights: class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """

        # FC layers to transform query and key
        processed_query = self.query_layer(query)
        processed_keys = self.memory_layer(keys)

        # scores: (b x t_q x t_k)
        scores = self.calc_score(processed_query, processed_keys)

        # Normalize the scores, softmax over t_k
        scores_normalized = F.softmax(scores, dim=-1)

        # Calculate the weighted average of the attention inputs according to
        # the scores
        # context: (b x t_q x n)
        context = torch.bmm(scores_normalized, keys)

        return context, scores_normalized


class MultiHeadAttention(nn.Module):
    """
        (Simplified) Multi-Head Attention (https://arxiv.org/abs/1706.03762)
        Implementation is very similar to https://github.com/KinglittleQ/GST-Tacotron
        """
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
                 dropout=0.5,
                 h=8,
                 is_masked=False):
        """

        :param query_dim: feature dimension for query
        :param key_dim: feature dimension for keys (memory)
        :param num_units: internal feature dimension
        :param dropout:
        :param h: times for concat dk = num_units / h
        :param is_masked:
        """
        super(MultiHeadAttention, self).__init__()

        if query_dim != key_dim:
            raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")

        self._num_units = num_units
        self._h = h
        self._key_dim = torch.tensor(key_dim, requires_grad=False).float()
        self._dropout = dropout
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim,num_units, bias=False)
        self.bn = nn.BatchNorm1d(num_units)
        self.ln = nn.LayerNorm(num_units)

    def forward(self, query, keys):
        processed_query = self.query_layer(query)
        processed_key = self.key_layer(keys)
        processed_value = self.value_layer(keys)

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        split_size = self._num_units // self.h
        processed_query = torch.cat(processed_query.split(split_size=split_size, dim=2), dim=0)
        processed_key = torch.cat(processed_key.split(split_size=split_size, dim=2), dim=0)
        processed_value = torch.cat(processed_value.split(split_size=split_size, dim=2), dim=0)

        # calculate QK^T
        attention = torch.matmul(processed_query, processed_key.transpose(1,2))
        # normalize with sqrt(dk)
        attention = attention / (self._key_dim ** 0.5)
        # use masking (usually for decoder) to prevent leftward
        # information flow and retains auto-regressive property
        # as said in the paper
        if self._is_masked:
            diag_vals = attention[0].sign().abs()
            diag_mat = diag_vals.tril()
            diag_mat = diag_mat.unsqueeze(0).expand(attention.size())
            mask = torch.ones(diag_mat.size()) * (-2 ** 32 + 1)
            # this is some trick that I use to combine the lower diagonal
            # matrix and its masking. (diag_mat-1).abs() will reverse the value
            # inside diag_mat, from 0 to 1 and 1 to zero. with this
            # we don't need loop operation andn could perform our calculation
            # faster
            attention = (attention * diag_mat) + (mask * (diag_mat - 1).abs())

        attention = F.softmax(attention, dim=-1)
        # apply dropout
        # attention = F.dropout(attention, self._dropout_p)
        # multiplyt it with V
        attention = torch.matmul(attention, processed_value)
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self._h)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=0), dim=2)

        return attention


class MatrixAttention(nn.Module):
    def __init__(self):
        super(MatrixAttention, self).__init__()


if __name__ == "__main__":
    ba = BahdanauAttention(256, 256, 256)
    query = torch.randn(5, 1, 256)
    keys = torch.randn(5, 5, 256)
    output, weights = ba(query, keys)
    print(output.shape)
    print(weights.shape)
