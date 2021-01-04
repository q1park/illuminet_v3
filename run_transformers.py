# limitations under the License.
import math
import collections
from dataclasses import dataclass
from typing import Union, Optional, Dict, Any, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader

from transformers import (
    XLNetModel, 
    XLNetPreTrainedModel, 
    BertPreTrainedModel,
    BertModel,
    T5PreTrainedModel,
    T5Model,
    AutoConfig,
    AutoTokenizer,
)

from transformers.modeling_utils import SequenceSummary
from transformers.models.xlnet.modeling_xlnet import XLNetForSequenceClassificationOutput
from transformers.models.bert.modeling_bert import SequenceClassifierOutput

from src.modeling.modules import AttentionPoolLayers, AttentionPool

class MyXLNetForSequenceClassification(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLNetModel(config)
        self.pool_layers = AttentionPoolLayers(768, 64, d_out=768)
        self.pool_words = AttentionPool(768, 128, d_out=128)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(3*128, config.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.init_weights()
        
    def embed(self, 
              input_ids=None,
              attention_mask=None,
              mems=None,
              perm_mask=None,
              target_mapping=None,
              token_type_ids=None,
              input_mask=None,
              head_mask=None,
              inputs_embeds=None,
              use_mems=None,
              output_attentions=None,
              output_hidden_states=None,
              return_dict=None,
              labels=None
             ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )
#         embedding = torch.cat([v.unsqueeze(-2) for v in transformer_outputs.hidden_states], dim=-2)
#         embedding = self.dropout(self.pool_layers(transformer_outputs))
        embedding = self.dropout(self.pool_words(transformer_outputs.last_hidden_state, attention_mask))
#         embedding = self.sequence_summary(transformer_outputs.last_hidden_state)
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
        **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in ``[0, ...,
            config.num_labels - 1]``. If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch1 = {
            'attention_mask':attention_mask[..., 0],
            'input_ids':input_ids[..., 0], 
            'token_type_ids':token_type_ids[..., 0],
            'labels':labels
        }
        
        batch2 = {
            'attention_mask':attention_mask[..., 1],
            'input_ids':input_ids[..., 1], 
            'token_type_ids':token_type_ids[..., 1],
            'labels':labels
        }

        embedding1, transformer_outputs = self.embed(**batch1,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        embedding2, transformer_outputs = self.embed(**batch2,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
#         logits = self.logits_proj(torch.cat([embedding1, embedding2], dim=-1))
        
        logits = self.logits_proj(torch.cat([embedding1, embedding2, torch.abs(embedding1-embedding2)], dim=-1))
        labels = batch1['labels']

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetForSequenceClassificationOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.last_hidden_state,
            attentions=transformer_outputs.attentions,
        )
    
class MyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.pool_layers = AttentionPoolLayers(768, 768, d_out=768)
        self.pool_words = AttentionPool(768, 128, d_out=128)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(3*768, config.num_labels)
        self.sequence_summary = SequenceSummary(config)
        
        self.init_weights()
        
    def embed(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        embedding = outputs[1]
#         embedding = torch.cat([v.unsqueeze(-2) for v in outputs.hidden_states], dim=-2)
#         embedding = self.dropout(self.pool_layers(embedding))
#         embedding = self.dropout(self.pool_words(outputs[0]))
        return embedding, outputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        idx=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch1 = {
            'attention_mask':attention_mask[..., 0],
            'input_ids':input_ids[..., 0], 
            'token_type_ids':token_type_ids[..., 0],
            'labels':labels
        }
        
        batch2 = {
            'attention_mask':attention_mask[..., 1],
            'input_ids':input_ids[..., 1], 
            'token_type_ids':token_type_ids[..., 1],
            'labels':labels
        }

        embedding1, outputs = self.embed(**batch1,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        embedding2, outputs = self.embed(**batch2,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
#         logits = self.logits_proj(torch.cat([embedding1, embedding2], dim=-1))
        
        embedding = torch.cat([embedding1, embedding2, torch.abs(embedding1-embedding2)], dim=-1)
        logits = self.classifier(embedding)
        labels = batch1['labels']
        
#         embedding, outputs = self.embed(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
        
#         embedding = self.dropout(embedding)
#         logits = self.classifier(embedding)
#         print(logits.shape)
    
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
from src.nlp.glue_trainer import MRPCTrainer

from transformers import set_seed
from transformers.trainer_utils import is_main_process

from src.nlp.glue_utils import (
    ModelArguments, 
    DataTrainingArguments, 
    TrainingArguments, 
    GluePreprocessor, 
    AutoPreprocessor,
    ComputeMetrics
)

from datasets import DatasetDict

model_args = ModelArguments(model_name_or_path='bert-base-cased')
data_args = DataTrainingArguments(task_name='mnli', max_seq_length=80)
training_args = TrainingArguments(
    do_train=True, do_eval=True, 
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5, num_train_epochs=3.0,
    output_dir='nli_output_poolstd',
    overwrite_output_dir=True,
    evaluation_strategy='steps',
    eval_accumulation_steps=1,
    logging_steps=500,
    eval_steps=500,
    local_rank=-1
)

set_seed(training_args.seed)

tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
)

config = AutoConfig.from_pretrained(
    model_args.model_name_or_path,
#     num_labels=preprocessor.num_labels,
    finetuning_task=data_args.task_name,
    cache_dir=model_args.cache_dir,
)

dataset_dict = DatasetDict()
dataset_dict = dataset_dict.load_from_disk('data/nli/mnli_snli_hans')
preprocessor = AutoPreprocessor(data_args, dataset_dict, config, tokenizer)

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

config.update({
    "num_labels":preprocessor.num_labels,
    "summary_activation": "tanh",
    "summary_last_dropout": 0.1,
    "summary_type": "mean",
    "summary_use_proj": True
})

model = MyBertForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
)

train_dataset, eval_dataset, test_dataset = dataset_dict['train'], dataset_dict['validation'], dataset_dict['test']

from transformers.file_utils import is_datasets_available

from transformers import default_data_collator

# Initialize our Trainer
trainer = MRPCTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=ComputeMetrics(data_args, preprocessor),
    tokenizer=preprocessor.tokenizer,
    data_collator=default_data_collator if data_args.pad_to_max_length else None,
)

# Training
if training_args.do_train:
    trainer.train()
#     trainer.train(train_dataset2=datasets2['train'])
    trainer.save_model()  # Saves the tokenizer too for easy upload