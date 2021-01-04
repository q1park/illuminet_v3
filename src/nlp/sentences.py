# limitations under the License.
"""
 PyTorch XLNet model.
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from transformers import XLNetConfig
from transformers.models.xlnet.modeling_xlnet import (
    XLNetModel, 
    XLNetPreTrainedModel, 
    XLNetForSequenceClassificationOutput
)

import math
import collections

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Union, Optional, Dict, Any, List, Tuple

from transformers import Trainer
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import DistributedTensorGatherer, nested_concat, nested_detach
from transformers.trainer_utils import EvalPrediction, PredictionOutput, TrainOutput, set_seed
from transformers.modeling_utils import SequenceSummary

class MyT5ForSequenceClassification(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = T5Model(config)
        self.pool_layers = AttentionPoolLayers(768, 64, d_out=768)
        self.pool_words = AttentionPool(768, 64, d_out=768)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(3*768, config.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.init_weights()
        
    def embed(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None
    ):
        transformer_outputs = self.transformer(
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            head_mask=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    )
#         embedding = torch.cat([v.unsqueeze(-2) for v in transformer_outputs.hidden_states], dim=-2)
#         embedding = self.dropout(self.pool_layers(transformer_outputs))
#         embedding = self.dropout(self.pool_words(transformer_outputs.last_hidden_state, attention_mask))
        embedding = self.sequence_summary(transformer_outputs.last_hidden_state)
        return embedding, transformer_outputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_mems=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,  # delete when `use_cache` is removed in T5Model
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in ``[0, ...,
            config.num_labels - 1]``. If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        nb = input_ids.size(0)

        return T5ForSequenceClassificationOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.last_hidden_state,
            attentions=transformer_outputs.attentions,
        )