import nltk

try:
    nltk.download('stopwords')
    nltk.download('wordnet')
except:
    print("Couldn't download stopwords or wordnet")

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from datasets import load_dataset
from transformers import BertTokenizerFast
import pandas as pd
from tqdm.auto import tqdm

from modeling_utils import Classifier
from data_utils import build_or_load_vocab, preprocess

# pass path to test data here
DATA_FILES = "test-v2.csv"

MODEL_ID = "vasudevgupta/amazon-ml-hack-best"
TOKENIZER_ID = "bert-base-uncased"


def _mapping_fn(sample, sep_token):
    if sample["BULLET_POINTS"] is None:
        sample["BULLET_POINTS"] = ""
    if sample["DESCRIPTION"] is None:
        sample["DESCRIPTION"] = ""
    if sample["TITLE"] is None:
        sample["TITLE"] = ""
    desc = sample["BULLET_POINTS"] + f" {sep_token} " + sample["DESCRIPTION"]
    sample["inputs"] = sample["TITLE"] + f" {sep_token} " + desc
    return sample


@jax.jit
def _forward(input_ids, attention_mask):
    logits = model(input_ids, attention_mask)[0]
    return jnp.argmax(logits, axis=-1)


def _predict(inputs, idx):
    outputs = tokenizer(inputs["inputs"], return_tensors="jax", max_length=512, truncation=True, padding="max_length")
    category = _forward(outputs["input_ids"], outputs["attention_mask"])
    inputs["BROWSE_NODE_ID"] = int(category.item())
    return inputs


@jax.jit
def _random_forward(input_ids, attention_mask, rng):
    logits = model(input_ids, attention_mask)[0]
    return jax.random.categorical(rng, logits, axis=-1)


def _random_predict(inputs, idx):
    outputs = tokenizer(inputs["inputs"], return_tensors="jax", max_length=512, truncation=True, padding="max_length")
    rng = jax.random.PRNGKey(idx)
    category = _random_forward(outputs["input_ids"], outputs["attention_mask"], rng)
    inputs["BROWSE_NODE_ID"] = int(category.item())
    return inputs


if __name__ == "__main__":
    print("Using", jax.devices())

    data = load_dataset("csv", data_files=[DATA_FILES], split="train")

    browse_node_vocab = build_or_load_vocab(column_name="BROWSE_NODE_ID")
    to_browse_node = {v: k for k, v in browse_node_vocab.items()}

    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_ID)

    data = data.map(_mapping_fn, fn_kwargs={"sep_token": tokenizer.sep_token})
    data = data.map(lambda x: {"len_inputs": len(x["inputs"]) // 4})
    print("data stats:", {
        "max": np.max(data["len_inputs"]),
        "mean": np.mean(data["len_inputs"]),
        "min": np.min(data["len_inputs"]),
    })

    model = Classifier.from_pretrained(MODEL_ID, num_browse_nodes=len(browse_node_vocab))

    # replace _predict with _random_predict if want to use jax.random.categorical
    data = data.map(_predict, with_indices=True)
    data = data.remove_columns(['TITLE', 'DESCRIPTION', 'BULLET_POINTS', 'BRAND', 'inputs', 'len_inputs'])
    print(data)

    print("saving `submission.csv`")
    data.to_csv("submission.csv")
