from typing import Dict, Optional

import torch
import torch.distributed as dist
from allennlp.common import util
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from transformers import RobertaTokenizer

from StD.common.masked_lm_utils import mask_tokens
from StD.common.contrastive_utils import mine_triplet
from StD.common.model_utils import unpack_batch
#from StD.losses import PyTorchMetricLearningLoss
#from StD.miners import PyTorchMetricLearningMiner
from StD.losses.triplet_loss import PyTorchMetricLearningTripletLoss
from StD.miners.triplet_miner import PyTorchMetricLearningTripletMiner
from StD.losses.custom_triplet_loss import *


@Model.register("StD")
class RStanceDet(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Optional[Seq2VecEncoder] = None,
        feedforward: Optional[FeedForward] = None,
        miner: Optional[PyTorchMetricLearningTripletMiner] = None,
        loss: Optional[PyTorchMetricLearningTripletLoss] = None,
        scale_fix: bool = True,
        triplet_mining_strategy: str = "random",
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        token_embedder = self._text_field_embedder._token_embedders["tokens"]
        self._masked_language_modeling = token_embedder.masked_language_modeling
        if self._masked_language_modeling:
            self._tokenizer = token_embedder.tokenizer

        # Default to mean BOW pooler. This performs well and so it serves as a sensible default.
        self._seq2vec_encoder = seq2vec_encoder or BagOfEmbeddingsEncoder(
            text_field_embedder.get_output_dim(), averaged=True
        )
        self._feedforward = feedforward
        self._miner = miner
        self._loss = loss
        self._triplet_mining_strategy = triplet_mining_strategy

        if self._loss is None and not self._masked_language_modeling:
            raise ValueError(("No loss function provided"))

        self._scale_fix = scale_fix
        initializer(self)

    def forward(self, text, labels) -> Dict[str, torch.Tensor]:

        output_dict: Dict[str, torch.Tensor] = {}

        if self.training and self._masked_language_modeling:
            text = mask_tokens(text, self._tokenizer)
        masked_lm_loss, embedded_text = self._forward_internal(text, output_dict)

        if self.training:
            output_dict["loss"] = 0
            if self._loss is not None:
                if self._triplet_mining_strategy == "random":
                    contrastive_loss = self._loss(embedded_text, labels)
                elif self._triplet_mining_strategy == "hard":
                    contrastive_loss = batch_hard_triplet_loss(labels, embedded_text, 0.5, True)
                elif self._triplet_mining_strategy == "hard_easy":
                    contrastive_loss = batch_hard_easy_triplet_loss(labels, embedded_text, 0.5, True)
                else:
                    print("Invalid triplet strategy")
                    raise ValueError(("Provid one of [random, hard, hard_easy] values for triplet strategy"))

                if util.is_distributed() and self._scale_fix:
                    contrastive_loss *= dist.get_world_size()
                output_dict["loss"] += contrastive_loss
            if masked_lm_loss is not None:
                output_dict["loss"] += masked_lm_loss

        return output_dict

    def _forward_internal(
        self,
        tokens: TextFieldTensors,
        output_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:

        masked_lm_loss, embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)
        # Don't hold on to embeddings or projections during training.
        if output_dict is not None and not self.training:
            output_dict["embeddings"] = embedded_text.clone().detach()
        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)
            if output_dict is not None and not self.training:
                output_dict["projections"] = embedded_text.clone().detach()

        return masked_lm_loss, embedded_text

    default_predictor = "StD"
