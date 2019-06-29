import torch
from torchtext import data
from copy import copy


class Dataset:

    def __init__(self,
                 data_dir="data",
                 train_name="train_data.tsv",
                 relation_name="newrelation.vocab",
                 mode="train",
                 device="cpu"):

        self.path = data_dir + "/" + train_name
        self.mode = mode
        print("Loading data from " + train_name)

        self.title = data.Field(sequential=True, batch_first=True, init_token="<start>", eos_token="<eos>",
                                include_lengths= True)
        self.label = data.Field(sequential=True, batch_first=True, init_token="<start>", eos_token="<eos>")
        self.entity = data.RawField()
        self.entity.is_target = False
        self.relation = data.RawField()
        self.relation.is_target = False
        self.output = data.Field(sequential=True, batch_first=True, init_token="<start>", eos_token="<eos>",
                                 include_lengths=True)
        self.node_order = data.RawField()
        self.node_order.is_target = False

        self.fields = [("title", self.title), ("entity", self.entity), ("label", self.label),
                       ("relation", self.relation), ("output", self.output), ("node_order", self.node_order)]
        train_data = data.TabularDataset(path=self.path, format="tsv", fields=self.fields)
        valid_data = data.TabularDataset(path=self.path.replace("train", "val"), format='tsv', fields=self.fields)

        self.target = data.Field(sequential=True, batch_first=True, init_token="<start>", eos_token="<eos>")

        print("Building Vocab...")
        self.output.build_vocab(train_data, min_freq=1)
        generics = ['<variable>', '<system>', '<interface>', '<hardware>', '<constant>', '<statevariable>',
                    '<message>', '<time>', '<state>', '<mode>', '<beacon>', '<cycle>', '<otheritem>']
        self.output.vocab.itos.extend(generics)
        self.target.vocab = copy(self.output.vocab)
        specials = "variable system interface hardware constant state variable " \
                   "message time state mode beacon cycle otheritem".split(" ")
        for x in specials:
            for y in range(40):
                s = "<" + x + "_" + str(y) + ">"
                self.target.vocab.stoi[s] = len(self.target.vocab.itos) + y
        self.label.build_vocab(train_data, min_freq=1)
        for x in generics:
            self.label.vocab.stoi[x] = self.output.vocab.stoi[x]

        self.title.build_vocab(train_data, min_freq=0)

        self.relation.special = ['<pad>', '<unk>', 'ROOT']
        with open(data_dir + "/" + relation_name) as f:
            _vocab = [x.strip() for x in f.readlines()]
            self.relation.size = len(_vocab)
            _vocab += [x + "_inv" for x in _vocab]
            relation_vocab = self.relation.special + _vocab
        self.relation.itos = relation_vocab

        self.entity.itos, self.entity.stoi = self.build_entity_vocab()

        print('done')
        if not mode == "eval":
            self.train_iter = self.make_iterator(train_data, mode="train", device=device)
            self.valid_iter = self.make_iterator(valid_data, mode="valid", device=device)

    def build_entity_vocab(self):
        ents = ""
        itos = list()
        itos.append("<unk>")
        itos.append("<pad>")
        with open(self.path) as f:
            for l in f:
                ents += " " + l.split("\t")[1]
        itos = sorted(list(set(ents.split(" "))))
        stoi = {x: i for i, x in enumerate(itos)}
        return itos, stoi

    def entity_to_vector(self):
        return

    def make_graphs(self, relation, entity_length):
        relation = relation.strip().split(";")
        relations = [[int(_) for _ in _rel.split()] for _rel in relation]
        adj_size = entity_length + 1 + 2*len(relations)
        adj_matrix = torch.zeros((adj_size, adj_size))
        for i in range(entity_length):
            adj_matrix[i, entity_length] = 1
            adj_matrix[entity_length, i] = 1
        for i in range(adj_size):
            adj_matrix[i][i] = 1
        _relations = [2]
        for rel in relations:
            # print(self.relation.itos)
            _relations.extend([rel[1] + 3, rel[1] + 3 + self.relation.size])
            a = rel[0]
            b = rel[2]
            c = entity_length + len(_relations) - 2
            d = entity_length + len(_relations) - 1
            adj_matrix[a, c] = 1
            adj_matrix[c, b] = 1
            adj_matrix[b, d] = 1
            adj_matrix[d, a] = 1
        _relations = torch.tensor(_relations)
        return adj_matrix, _relations

    def make_vocab(self):
        return

    def make_iterator(self, raw_data, mode, device):
        raw_data = data.Dataset(raw_data, fields=self.fields)
        for instance in raw_data:
            instance.entity = instance.entity.split(";")
            instance.relation = self.make_graphs(instance.relation, len(instance.entity))
            instance.target = instance.output
            instance.output = [_.split("_")[0]+">" if "_" in _ else _ for _ in instance.output]
            instance.node_target = torch.tensor([int(_) + 3 for _ in instance.node_order.split(" ")])  # ????
            instance.nodes = [[int(_) for _ in order.strip().split(" ")]
                               for order in instance.node_order.split("-1")[:-1]]
        if mode == "train":
            iterator = data.Iterator(raw_data, batch_size=16, device=device, sort_key=lambda x: len(x.out),
                                     repeat=False, train=True)
        else:
            iterator = data.Iterator(raw_data, batch_size=16, device=device, sort_key=lambda x: len(x.out),
                                     sort=False, repeat=False, train=False)
        return iterator

    def make_test(self, args):
        # To be implemented
        return


if __name__ == '__main__':
    dataset = Dataset(data_dir="data", train_name="train_data.tsv",relation_name="newrelation.vocab", mode="train")
    # dataset.make_graphs(dataset.relation, dataset.entity)
