import torch
from pytorch_metric_learning.miners import BaseTupleMiner
from allennlp.common import Registrable


class PyTorchMetricLearningTripletMiner(Registrable):
    default_implementation = "miner"


@PyTorchMetricLearningTripletMiner.register("miner")
class TripletMiner(PyTorchMetricLearningTripletMiner, BaseTupleMiner):

    def __init__(self, margin=0.1, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        group_one_indices = (ref_labels == 0).nonzero(as_tuple=True)
        group_two_indices = (ref_labels == 1).nonzero(as_tuple=True)

        group_one = torch.index_select(ref_emb, 1, group_one_indices[0])
        group_two = torch.index_select(ref_emb, 1, group_two_indices[0])

        print(group_two_indices)

        rand_indices_group1 = torch.randint(0, group_one_indices[0].size()[0], (5, ))
        rand_indices_group2 = torch.randint(0, group_two_indices[0].size()[0], (5, ))

        group_one = torch.index_select(group_one, 1, rand_indices_group1)
        group_two = torch.index_select(group_two, 1, rand_indices_group2)

        group_one_labels = torch.zeros(5)
        group_two_labels = torch.ones(5)
        group_one_two = torch.cat((group_one[0], group_two[0]))
        group_one_two_labels = torch.cat((group_one_labels, group_two_labels))

        indices_anchor = torch.arange(0, embeddings.size()[0])
        indices_group_one = torch.arange(0, group_one_labels.size()[0])
        indices_group_two = torch.arange(group_one_labels.size()[0], group_one_two_labels.size()[0])

        return ((indices_anchor, indices_group_one, indices_group_two))

