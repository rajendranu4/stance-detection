import logging
import random
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, ListField, TextField, ArrayField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, SpacyTokenizer, Tokenizer
from overrides import overrides
from transformers import RobertaTokenizer

from StD.common.util import sanitize_text
import pandas as pd

logger = logging.getLogger(__name__)


@DatasetReader.register("StD")
class StanceDatasetReader(DatasetReader):

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        num_anchors: int = None,
        num_positives: int = None,
        max_span_len: int = None,
        min_span_len: int = None,
        sampling_strategy: str = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        data = pd.read_csv(file_path)
        data = data.sample(frac=1)

        for _, text_label_row in data.iterrows():
            yield self.text_to_instance(text_label_row)

    @overrides
    def text_to_instance(self, text_label_row) -> Instance:  # type: ignore

        text = sanitize_text(text_label_row['text'], lowercase=False)

        fields = {}
        if isinstance(self._tokenizer, PretrainedTransformerTokenizer):
            text = f" {text.lstrip()}"
            tokenization_func = self._tokenizer.tokenizer.tokenize
            if self._tokenizer.tokenizer.is_fast:
                self._tokenizer.tokenizer._tokenizer.no_truncation()
        else:
            tokenization_func = None

        tokenized_text = self._tokenizer.tokenizer.decode(
            self._tokenizer.tokenizer.convert_tokens_to_ids(text.split())
        )

        tokenized_text = text

        tokens = self._tokenizer.tokenize(tokenized_text)
        tokens_textfield = TextField(tokens, self._token_indexers)

        fields["text"] = tokens_textfield
        fields["labels"] = LabelField(text_label_row['label'], skip_indexing=True)

        return Instance(fields)