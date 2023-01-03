from allennlp.common import Registrable

from StD.miners.triplet_miner import PyTorchMetricLearningTripletMiner
from pytorch_metric_learning import losses


class PyTorchMetricLearningTripletLoss(Registrable):
    default_implementation = "triplet_margin_loss"


@PyTorchMetricLearningTripletLoss.register("cross_batch_memory")
class CrossBatchMemory(PyTorchMetricLearningTripletLoss, losses.CrossBatchMemory):
    def __init__(
            self,
            loss: PyTorchMetricLearningTripletLoss,
            embedding_size: int,
            memory_size: int = 1024,
            miner: PyTorchMetricLearningTripletMiner = None,
    ) -> None:
        super().__init__(
            loss=loss,
            embedding_size=embedding_size,
            memory_size=memory_size,
            miner=miner,
        )


@PyTorchMetricLearningTripletLoss.register("triplet_margin_loss")
class TripletMargLoss(PyTorchMetricLearningTripletLoss, losses.TripletMarginLoss):
    def __init__(self,
                 margin: float,
                 swap: bool = False,
                 smooth_loss: bool = False,
                 triplets_per_anchor: str = 'all'
                 ) -> None:
        super().__init__(margin=margin,
                         swap=swap,
                         smooth_loss=smooth_loss,
                         triplets_per_anchor=triplets_per_anchor)
