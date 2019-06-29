import argparse
import torch
import torch.nn as nn
import torch.utils.data as data

from dataset import Dataset

from models.graph_attention import GraphEncoder


if __name__ == "__main__":
    dataset = Dataset(data_dir="data", train_name="train_data.tsv", relation_name="newrelation.vocab", mode="train")
    graph_encoder = GraphEncoder(64,64)
    for iter in dataset.train_iter:
        for (_, i) in enumerate(iter.dataset):
            adj_matrix, relation = i.relation
            entity = i.entity
            print(entity)
            graph_encoder.forward(adj_matrix, relation, entity, entity[2])
