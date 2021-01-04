# from src.nlp.glue_utils import AutoPreprocessor

# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric, Dataset

import transformers
from transformers import (
    XLNetConfig,
    XLNetTokenizer,
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

def get_labels(data_args, datasets):
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    return label_list, num_labels, is_regression

def get_label_to_id(data_args, config, num_labels, is_regression):
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in config.label2id.items()}

        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            print(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )

    elif data_args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}
    return label_to_id, padding, max_length

def split_datasets(data_args, datasets):
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]

    if data_args.task_name is not None:
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
    return train_dataset, eval_dataset, test_dataset

class ComputeMetrics:
    def __init__(self, data_args, processor):
        self.data_args = data_args
        self.processor = processor
        self.metric = load_metric("glue", data_args.task_name)
        
    def __call__(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if self.processor.is_regression else np.argmax(preds, axis=1)
        if self.data_args.task_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif self.processor.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
        
class GluePreprocessor:
    def __init__(self, model_args, data_args, datasets):
        self.label_list, self.num_labels, self.is_regression = get_labels(data_args, datasets)
        self.sentence1_key, self.sentence2_key = get_sentence_keys(data_args, datasets)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
        
        self.config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
        
        self.label_to_id, self.padding, self.max_length = get_label_to_id(
            data_args, self.config, self.num_labels, self.is_regression
        )
        
    def __call__(self, examples):
        args = (
            (examples[self.sentence1_key],) 
            if self.sentence2_key is None else 
            (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if self.label_to_id is not None and "label" in examples:
            result["label"] = [self.label_to_id[l] for l in examples["label"]]
        return result
    
class AutoPreprocessor:
    def __init__(self, data_args, dataset_dict, config, tokenizer):
        self.label_list, self.num_labels, self.is_regression = get_labels(data_args, dataset_dict)
        self.label_to_id, self.padding, self.max_length = get_label_to_id(
            data_args, config, self.num_labels, self.is_regression
        )
        self.tokenizer = tokenizer
        self.s1, self.s2 = 'sentence1', 'sentence2'
        
    def tokenize_process(self, sentence):
        return {
            k:np.expand_dims(np.array(v, dtype=np.int64), axis=-1)
            for k,v in self.tokenizer(
                sentence, padding=self.padding, max_length=self.max_length, truncation=True
            ).items()
        }
        
    def __call__(self, examples):
        result = self.tokenize_process(examples[self.s1])

        for k,v in self.tokenize_process(examples[self.s2]).items():
            result[k] = np.concatenate([result[k], v], axis=-1)
            
        if self.label_to_id is not None and "label" in examples:
            result["label"] = [self.label_to_id[l] for l in examples["label"]]
        return result
