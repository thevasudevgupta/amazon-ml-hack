from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase

import os
import jax.numpy as jnp
from tqdm.auto import tqdm
import json


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 128

    def __call__(self, batch):
        return self.collate_fn(batch)

    def collate_fn(self, features):
        # no dynamic padding on TPUs
        inputs = self.tokenizer(
            features["inputs"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="jax",
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "browse_nodes": jnp.array(features["BROWSE_NODE_ID"], dtype=jnp.int32),
            # "brands": jnp.array(features["BRAND"], dtype=jnp.int32),
        }


def batchify(dataset, batch_size, seed=None):
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    # extra samples will be dropped
    for i in range(len(dataset) // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size]
        yield dict(batch)


def preprocess(dataset, sep_token):
    dataset = dataset.map(_mapping_fn, fn_kwargs={"sep_token": sep_token})
    return dataset


def _mapping_fn(sample, sep_token):
    if sample["BULLET_POINTS"] is None:
        sample["BULLET_POINTS"] = ""
    if sample["DESCRIPTION"] is None:
        sample["DESCRIPTION"] = ""
    if sample["TITLE"] is None:
        sample["TITLE"] = ""
    desc = sample["BULLET_POINTS"] or sample["DESCRIPTION"]
    sample["inputs"] = sample["TITLE"] + f" {sep_token} " + desc
    return sample


def build_or_load_vocab(dataset=None, column_name="BROWSE_NODE_ID"):
    os.makedirs("../assets", exist_ok=True)
    if f"{column_name}.json" not in os.listdir("../assets"):
        assert dataset is not None
        print(f"building vocab from dataset[{column_name}]", end=" ... ")
        ids = set()
        for sample in tqdm(dataset):
            ids.update([sample[column_name]])
        vocab = {id: i for i, id in enumerate(ids)}
        with open(f"../assets/{column_name}.json", "w") as f:
            json.dump(vocab, f)
        print("done!!")
    else:
        with open(f"../assets/{column_name}.json") as f:
            vocab = json.load(f)
    return vocab
