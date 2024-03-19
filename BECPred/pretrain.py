import os
import numpy as np
import pandas as pd
import torch
import logging
import random
from rxnfp.models import SmilesLanguageModelingModel

logger = logging.getLogger(__name__)

config = {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.2,
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 512,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 4,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
"type_vocab_size": 2,
}
vocab_path = '../data/vocab.txt'

args = {'config': config, 
        'vocab_path': vocab_path, 
        'train_batch_size': 32,
        'manual_seed': 42,
        "fp16": False,
        "num_train_epochs": 12,
        'max_seq_length': 256,
        'evaluate_during_training': True,
        'overwrite_output_dir': True,
        'output_dir': '../models/pretrain',
        'learning_rate': 1e-4
       }


# optional
model = SmilesLanguageModelingModel(model_type='bert', model_name=None, args=args, use_cuda=True)

train_file = '../data/mlm_train_file.txt'
eval_file = '../data/mlm_eval_file_1k.txt'
model.train_model(train_file=train_file, eval_file=eval_file)
