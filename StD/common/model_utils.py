from typing import Tuple

import torch
import torch.distributed as dist
from allennlp.common import util
from allennlp.data import TextFieldTensors


def unpack_batch(tokens: TextFieldTensors) -> TextFieldTensors:
    """If the tensors of `tokens` are three-dimensional, we reshape them to be two-dimensional
    before returning the `TextFieldTensors` object. Otherwise, this is a no-op.

    # Parameters

    tokens : `TextFieldTensors`
        A `TextFieldTensors` object containnig the tensors to (possibly) reshape.

    # Returns

    `TextFieldTensors`
        Containing the (possibly) reshaped tensors.
    """
    for name, tensor in tokens["tokens"].items():
        if len(tensor.size()) == 3:
            tokens["tokens"][name] = tensor.reshape(tensor.size(0) * tensor.size(1), tensor.size(2))
    return tokens
