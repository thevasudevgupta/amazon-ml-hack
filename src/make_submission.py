import argparse
from functools import partial

import jax
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from modeling_utils import Classifier
from training_utils import predict
from data_utils import build_or_load_vocab, preprocess

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--data_files", default="../dataset/test-v2.csv", type=str)
    return parser

# python3 make_submission.py --model_id training-expt/training-expt/checkpoints-e0-s5640


if __name__ == '__main__':
    args = get_parser().parse_args()
    data = load_dataset("csv", data_files=[args.data_files], split="train")

    data = data.select(range(10))

    browse_node_vocab = build_or_load_vocab(column_name="BROWSE_NODE_ID")

    model = Classifier.from_pretrained(args.model_id, num_browse_nodes=len(browse_node_vocab))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data = preprocess(data, tokenizer.sep_token)

    data = data.map(lambda x: {"len_inputs": len(x["inputs"]) // 4})
    print("data stats:", {
        "max": np.max(data["len_inputs"]),
        "mean": np.mean(data["len_inputs"]),
        "min": np.min(data["len_inputs"]),
    })

    prediction_fn = partial(predict,
        forward_fn=jax.jit(model.__call__),
        to_browse_node={v: k for k, v in browse_node_vocab.items()},
        tokenizer=tokenizer,
        max_length=256,
    )

    data = data.map(prediction_fn)
    data = data.remove_columns(["TITLE", "DESCRIPTION", "BULLET_POINTS", "BRAND"])

    print("making `submission.csv`")
    data.to_csv("submission.csv")
