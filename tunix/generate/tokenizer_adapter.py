# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapt tokenizers to a common interface."""

import enum
import inspect
from typing import Any

from etils import epath
import numpy as np
import transformers

import sentencepiece as spm


class TokenizerType(enum.Enum):
  SP: str = 'sp'  # sentencepiece tokenizer
  HF: str = 'hf'  # huggingface tokenizer
  NONE: str = 'none'  # Represents no tokenizer


class TokenizerAdapter:
  """Wrapper for different tokenizers used in sampler."""

  def __init__(self, tokenizer: Any):
    self._tokenizer = tokenizer

    missing_methods = self._missing_methods()

    if isinstance(self._tokenizer, spm.SentencePieceProcessor):
      self._tokenizer_type = TokenizerType.SP
    elif self._is_hf_tokenizer():
      self._tokenizer_type = TokenizerType.HF
    elif not missing_methods:
      self._tokenizer_type = TokenizerType.NONE
    else:
      raise ValueError(
          'Your tokenizer should either be a `spm.SentencePieceProcessor` '
          'tokenizer, a HuggingFace tokenizer, or it should have '
          'the following methods: '
          '`["encode", "decode", "bos_id", "eos_id", "pad_id"]`. Received: '
          f'`type(tokenizer)` = {type(tokenizer)}, with missing methods: '
          f'{missing_methods}.'
      )

  def encode(self, text: str, **kwargs) -> list[int]:
    if self._tokenizer_type == TokenizerType.SP:
      return self._tokenizer.EncodeAsIds(text, **kwargs)
    elif self._tokenizer_type == TokenizerType.HF:
      return self._tokenizer.encode(text, **kwargs)
    else:
      return self._tokenizer.encode(text, **kwargs)

  def decode(self, ids: list[int], **kwargs) -> str:
    if self._tokenizer_type == TokenizerType.SP:
      return self._tokenizer.DecodeIds(ids, **kwargs)
    elif self._tokenizer_type == TokenizerType.HF:
      return self._tokenizer.decode(ids, **kwargs)
    else:
      return self._tokenizer.decode(ids, **kwargs)

  def bos_id(self) -> int:
    if self._tokenizer_type == TokenizerType.SP:
      return self._tokenizer.bos_id()
    elif self._tokenizer_type == TokenizerType.HF:
      return self._tokenizer.bos_token_id
    else:
      return self._tokenizer.bos_id()

  def eos_id(self) -> int:
    if self._tokenizer_type == TokenizerType.SP:
      return self._tokenizer.eos_id()
    elif self._tokenizer_type == TokenizerType.HF:
      return self._tokenizer.eos_token_id
    else:
      return self._tokenizer.eos_id()

  def pad_id(self) -> int:
    """Returns the pad token id."""
    if self._tokenizer_type == TokenizerType.SP:
      ret_id = self._tokenizer.pad_id()
      if ret_id == -1:
        raise ValueError('SentencePiece tokenizer has a undefined pad_id.')
      return ret_id
    elif self._tokenizer_type == TokenizerType.HF:
      # e.g. llama3 HF tokenizers do not have pad_id
      if self._tokenizer.pad_token_id is None:
        self._tokenizer.pad_token = self._tokenizer.eos_token
        # self._tokenizer.add_special_tokens({"pad_token": "<pad>"})
      return self._tokenizer.pad_token_id
    else:
      return self._tokenizer.pad_id()

  def _missing_methods(self) -> list[str]:
    """Checks if the tokenizer has any missing methods."""
    required_methods = ['encode', 'decode', 'bos_id', 'eos_id', 'pad_id']
    missing_methods = []
    for method in required_methods:
      if not hasattr(self._tokenizer, method):
        missing_methods.append(method)
    return missing_methods

  def _is_hf_tokenizer(self) -> bool:
    """Checks if the tokenizer is a huggingface tokenizer."""
    baseclasses = inspect.getmro(type(self._tokenizer))
    baseclass_names = [
        baseclass.__module__ + '.' + baseclass.__name__
        for baseclass in baseclasses
    ]
    if (
        'transformers.tokenization_utils_base.PreTrainedTokenizerBase'
        in baseclass_names
    ):
      return True
    return False

  @property
  def tokenizer(self) -> Any:
    return self._tokenizer


class Tokenizer(TokenizerAdapter):
  """Tokenizing and encoding/decoding text using TokenizerAdapter."""

  def __init__(
      self,
      tokenizer_type: str = 'sentencepiece',
      tokenizer_path: str = 'gs://gemma-data/tokenizers/tokenizer_gemma2.model',
      add_bos: bool | None = True,
      add_eos: bool | None = True,
      hf_access_token: str | None = None,
  ):

    self.tokenizer_type = tokenizer_type
    if tokenizer_type == 'huggingface':
      tokenizer = transformers.AutoTokenizer.from_pretrained(
          pretrained_model_name_or_path=tokenizer_path,
          add_bos_token=add_bos,
          add_eos_token=add_eos,
          token=hf_access_token,
          use_fast=True,
      )
    elif tokenizer_type == 'sentencepiece':
      model_proto = epath.Path(tokenizer_path).read_bytes()
      tokenizer = spm.SentencePieceProcessor()
      tokenizer.LoadFromSerializedProto(model_proto)
      options = []
      if add_bos:
        options.append('bos')
      if add_eos:
        options.append('eos')

      extra_options_str = ':'.join(options)
      if extra_options_str:
        tokenizer.SetEncodeExtraOptions(extra_options_str)
    else:
      raise ValueError(f'Unsupported tokenizer_type: {tokenizer_type}')
    super().__init__(tokenizer)

  def tokenizasde(
      self,
      example: str,
      prefix: str = '',
      suffix: str = '',
      add_eos: bool = True,
      max_len = -1,
  ) -> np.ndarray:
    """The tokenization function.

    Args:
      example: Input string to tokenize.
      prefix:  Prefix to add to the input string.
      suffix:  Suffix to add to the input string.
      add_eos: If True, add an "end of sentence" token at the end of the output
        sequence.

    Returns:
      Tokens corresponding to the input string.
    """
    assert max_len > 0
    int_list = []
    if self.bos_id():
      int_list.append(self.bos_id())
    if self.tokenizer_type == 'huggingface':
      int_list.extend(
          self.encode(prefix + example + suffix, add_special_tokens=False)
      )
    else:
      # sentencepiece
      int_list.extend(self.tokenizer.EncodeAsIds(prefix + example + suffix))
    if add_eos:
      int_list = int_list[:max_len-1]
      int_list.append(self.eos_id())
    else:
      int_list = int_list[:max_len]
    return np.array(int_list, dtype=np.int32)

  def tokenize(
      self,
      example: str,
      prefix: str = '',
      suffix: str = '',
      add_eos: bool = True,
  ) -> np.ndarray:
    """The tokenization function.

    Args:
      example: Input string to tokenize.
      prefix:  Prefix to add to the input string.
      suffix:  Suffix to add to the input string.
      add_eos: If True, add an "end of sentence" token at the end of the output
        sequence.

    Returns:
      Tokens corresponding to the input string.
    """
    int_list = []
    if self.bos_id():
      int_list.append(self.bos_id())
    if self.tokenizer_type == 'huggingface':
      int_list.extend(
          self.encode(prefix + example + suffix, add_special_tokens=False)
      )
    else:
      # sentencepiece
      int_list.extend(self.tokenizer.EncodeAsIds(prefix + example + suffix))
    if add_eos:
      int_list.append(self.eos_id())
    return np.array(int_list, dtype=np.int32)


  # def tokenize_batch(self, examples, prefix='', suffix='', add_eos=True, max_len=-1):
  #   assert max_len > 0

  #   def _pad_up_to_max_len(arr, pad_value, max_len):
  #     seq_len = arr.shape[0]
  #     to_pad = np.maximum(max_len - seq_len, 0)
  #     return np.pad(arr, [[0, to_pad]], mode='constant', constant_values=pad_value)

  #   pad_value = self.pad_id()
  #   seqs = [self.tokenize(s, prefix, suffix, add_eos, max_len) for s in examples]
  #   L = max_len if max_len > 0 else max(len(x) for x in seqs)
  #   toks = [_pad_up_to_max_len(x, pad_value, L) for x in seqs]
  #   return np.stack(toks)

  def tokenize_batch(self, examples, prefix='', suffix='', add_eos=True, max_len=-1, force_eos=True):
    assert max_len > 0
    pad_value = self.pad_id()
    eos_id = self.eos_id()
    pad_id = pad_value

    if self.tokenizer_type == 'huggingface':
      texts = [prefix + s + suffix for s in examples]
      out = self.tokenizer(
        texts,
        add_special_tokens=add_eos,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_attention_mask=False,
        return_tensors="np",
      )
      ids = out["input_ids"]          # (B, L)
      B, L = ids.shape

      # # mask of pads
      # pad_mask = (ids == pad_id)      # (B, L)
      # has_pad = pad_mask.any(axis=1)  # (B,)

      # # index of first pad in each row; if no pad, treat as L
      # first_pad_idx = np.where(
      #     has_pad,
      #     pad_mask.argmax(axis=1),    # first True along axis=1
      #     L,
      # )  # shape (B,)

      # # EOS position: one before first pad, or last position if no pad
      # eos_pos = np.where(first_pad_idx > 0, first_pad_idx - 1, L - 1)

      # rows = np.arange(B)
      # ids[rows, eos_pos] = eos_id     # put EOS at last non-pad token

      # # attention_mask is still correct:
      # # we only changed the *id* of an already non-pad position.

      # Always put EOS at the last position
      if force_eos:
        ids[:, -1] = eos_id

      out["input_ids"] = ids

      # print("out", out)
      return out["input_ids"]

    def _pad_up_to_max_len(arr, pad_value, max_len):
      seq_len = arr.shape[0]
      to_pad = np.maximum(max_len - seq_len, 0)
      return np.pad(arr, [[0, to_pad]], mode='constant', constant_values=pad_value)

    seqs = [self.tokenize(s, prefix, suffix, add_eos, max_len) for s in examples]
    L = max_len if max_len > 0 else max(len(x) for x in seqs)
    toks = [_pad_up_to_max_len(x, pad_value, L) for x in seqs]
    return np.stack(toks)