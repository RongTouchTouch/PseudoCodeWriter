import torch
from torchtext import data


class Dataset:

    def __init__(self, data_dir, filename): #args datadir, data
        self.path = data_dir + filename
        print("Loading data from" + filename)
        self.title = data.Field(sequential=True, batch_first=True, init_token="<start>", eos_token="<eos>",
                                include_lengths= True)
        self.label = data.Field(sequential=True, batch_first=True, init_token="<start>", eos_token="<eos>")
        self.entity  = data.RawField()
        self.relation = data.RawField()
        self.output = data.Field(sequential=True, batch_first=True, init_token="<start>", eos_token="<eos>",
                                include_lengths= True)
        self.node_order = data.RawField()

        self.fields = [("title", self.title), ("entity", self.entity), ("label", self.label),
                       ("relation", self.relation), ("output", self.output),("node_order", self.node_order)]


    def build_entity_vocab(self):


    def entity_to_vector(self):


    def make_graphs(self):
        dataset = data.TabularDataset(path=self.path, format="tsv", fields=self.fields)

        print("Building Vocab")

        self.output.build_vocab(dataset, minfreq = 5)




    def make_vocab(self):



