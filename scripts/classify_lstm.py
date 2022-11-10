#!/usr/bin/env python
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
import sys
import os
import time

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

sys.path.append(os.getcwd())
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import handystuff.logs
import numpy as np
import sklearn.metrics
import torch.autograd
import wandb
from datasets import load_dataset, load_metric, concatenate_datasets

from transformers import (
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
from nlr.util import get_args, dump_args

check_min_version("4.5.0")


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    from_scratch: bool = field(default=False, metadata={"help": "Whether to train from scratch."})

    max_seq_length: int = field(
        default=512,
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
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    hp_sweep: bool = field(default=False, metadata={"help": "Whether to do a hp sweep."})
    inoculate_eval: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

    sentence_col: Optional[str] = field(default=None, metadata={"help": "Name of column to use as input"})

    target_col: str = field(default='label', metadata={"help": "Name of column to use as label"})

    scheduler_step: int = 50

    def __post_init__(self):
        if self.train_file is None or self.validation_file is None:
            raise ValueError("Need training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name: Optional[str] = field(
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
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )

    hidden_size: int = field(
        default=1024,
        metadata={'help': 'Hidden dimensionality.'}
    )
    num_attention_heads: int = field(
        default=16,
        metadata={'help': 'Number of attention heads.'}
    )
    num_hidden_layers: int = field(
        default=24,
        metadata={'help': 'Number of attention layers.'}
    )
    init_xavier: bool = False
    dropout: float = 0.95
    hidden_size_ffnn = 300
    activation_fn = 'relu'


class BinaryLSTMClassifier(nn.ModuleList):

    def __init__(self, model_args: ModelArguments, vocab_size):
        super(BinaryLSTMClassifier, self).__init__()

        self.hidden_dim = model_args.hidden_size
        self.layers = model_args.num_hidden_layers
        self.input_size = vocab_size + 1  # Plus padding
        self.initialise_xavier = model_args.init_xavier
        self.hidden_dim_ffnn = model_args.hidden_size_ffnn
        self.dropout = model_args.dropout
        self.activation = getattr(F, model_args.activation_fn)
        self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.layers,
                            batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim_ffnn)
        self.fc2 = nn.Linear(self.hidden_dim_ffnn, 1)

    def forward(self, x, lengths, **kwargs):

        #seq_unpacked, lens_unpacked = pad_packed_sequence(x, batch_first=True, padding_value=self.embedding.padding_idx)
        out = self.embedding(x)
        out = pack_padded_sequence(out, lengths, batch_first=True, enforce_sorted=False)
        if self.initialise_xavier:
            h = torch.zeros((self.layers, x.size(0), self.hidden_dim))
            c = torch.zeros((self.layers, x.size(0), self.hidden_dim))

            torch.nn.init.xavier_normal_(h)
            torch.nn.init.xavier_normal_(c)

            out, (hidden, cell) = self.lstm(out, (h, c))
        else:

            out, (h, c) = self.lstm(out)
            # print(out.shape)
        out = F.dropout(h[-1], p=self.dropout)
        out = self.activation(self.fc1(out))
        out = F.dropout(out, p=self.dropout)
        out = torch.sigmoid(self.fc2(out))
        return out.squeeze()


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    torch.autograd.set_detect_anomaly(True)

    model_args, data_args, training_args = get_args(ModelArguments, DataTrainingArguments, TrainingArguments)
    data_args: DataTrainingArguments
    model_args: ModelArguments
    training_args: TrainingArguments
    wandb.init(config=training_args, name=training_args.run_name)
    if data_args.hp_sweep:

        for k, v in dict(wandb.config).items():
            original_arg = getattr(training_args, k)
            if original_arg != v and not isinstance(original_arg, Enum):
                setattr(training_args, k, v)
                print(f"SETTING {k} from {original_arg} to {v}!!!!11111")
        training_args.run_name = None
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    # )
    handystuff.logs.setup_logging()
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

    # Get the test dataset: you can provide your own CSV/JSON test file (see below)
    # when you use `do_predict` without specifying a GLUE benchmark task.
    if training_args.do_predict:
        if data_args.test_file is not None:
            train_extension = data_args.train_file.split(".")[-1]
            test_extension = data_args.test_file.split(".")[-1]
            assert (
                    test_extension == train_extension
            ), "`test_file` should have the same extension (csv or json) as `train_file`."
            data_files["test"] = data_args.test_file
        else:
            raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

    if data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        datasets = load_dataset("csv", data_files=data_files)
    else:
        # Loading a dataset from local json files
        datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    label_list = datasets["train"].unique(data_args.target_col)
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    def yield_tokens(data_iter):
        for t in data_iter:
            yield tokenizer(t['sentence'])

    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens((x for x in datasets['train'])), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: 1 if x == 'consistent' else 0

    # TODO: define model

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for item in batch:
            _label, _text = item['label'], item['sentence']
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))

        label_list = torch.tensor(label_list, dtype=torch.int64)

        if model_args.model_name == 'bow':
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
            text_list = torch.cat(text_list)
            return label_list.to(training_args.device), text_list.to(training_args.device), offsets.to(
                training_args.device)
        else:
            inputs = pad_sequence(text_list, batch_first=True)
            # inputs = pack_sequence(text_list, enforce_sorted=False)
            return label_list.to(training_args.device), inputs.to(training_args.device), offsets[1:]

    dataloader = DataLoader(iter(x for x in datasets['train']), batch_size=training_args.per_device_train_batch_size,
                            shuffle=False, collate_fn=collate_batch)

    vocab_size = len(vocab)
    model = (TextClassificationModel(vocab_size, model_args.hidden_size, num_labels)
             if model_args.model_name == 'bow'
             else BinaryLSTMClassifier(model_args, vocab_size)).to(training_args.device)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        print(preds)
        print(p.label_ids)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item(),
                "mcc": sklearn.metrics.matthews_corrcoef(preds, p.label_ids)}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.

    # Hyperparameters

    criterion = torch.nn.BCELoss() if isinstance(model, BinaryLSTMClassifier) else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=training_args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, data_args.scheduler_step, gamma=0.1)
    total_accu = None
    train_iter, test_iter = iter(x for x in datasets['train']), iter(x for x in datasets['validation'])
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_dataloader = DataLoader(split_train_, batch_size=training_args.per_device_train_batch_size,
                                  shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=training_args.per_device_eval_batch_size,
                                  shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=training_args.per_device_eval_batch_size,
                                 shuffle=True, collate_fn=collate_batch)

    def train(dataloader):
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()
        total_loss = 0

        for idx, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label if model_args.model_name == 'bow' else label.float())
            total_loss += loss
            loss.backward()
            if training_args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
            optimizer.step()
            if model_args.model_name == 'bow':
                total_acc += (predicted_label.argmax(1) == label).sum().item()
            else:
                total_acc += (predicted_label.round() == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                                  total_acc / total_count))
                total_acc, total_count = 0, 0
            start_time = time.time()
        elapsed = time.time() - start_time
        print('| epoch {:3d} | lr {} | train accuracy {:8.3f} | loss {:5f}'.format(epoch, scheduler.get_lr(),
                                                                                   total_acc / total_count, total_loss))
        total_loss = 0

    def evaluate(dataloader):
        model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = model(text, offsets)
                loss = criterion(predicted_label, label if model_args.model_name == 'bow' else label.float())
                if model_args.model_name == 'bow':
                    total_acc += (predicted_label.argmax(1) == label).sum().item()
                else:
                    total_acc += (predicted_label.round() == label).sum().item()
                total_count += label.size(0)
        return total_acc / total_count

    # Training
    if training_args.do_train:
        for epoch in range(1, int(training_args.num_train_epochs) + 1):
            epoch_start_time = time.time()
            train(train_dataloader)
            accu_val = evaluate(valid_dataloader)
            if total_accu is not None and total_accu > accu_val:
                # scheduler.step()
                ...
            else:
                total_accu = accu_val
            scheduler.step()
            print('-' * 59)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                  'valid accuracy {:8.3f} '.format(epoch,
                                                   time.time() - epoch_start_time,
                                                   accu_val))
            print('-' * 59)
    # Evaluation
    if training_args.do_eval:
        print('Checking the results of test dataset.')
        accu_test = evaluate(test_dataloader)
        print('test accuracy {:8.3f}'.format(accu_test))
    os.makedirs(training_args.output_dir, exist_ok=True)
    fn = os.path.join(training_args.output_dir, f"experiment-args.json")
    dump_args(model_args, data_args, training_args, file_name=fn)
    wandb.save(fn)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
