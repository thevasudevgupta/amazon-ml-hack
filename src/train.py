import os
from dataclasses import dataclass, asdict, replace
from functools import partial

import wandb
import jax
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from data_utils import DataCollator, batchify, build_or_load_vocab, preprocess
from modeling_utils import Classifier
from training_utils import (
    Trainer,
    build_tx,
    cls_loss_fn,
    train_step,
    val_step,
)

IGNORE_IDX = -100

@dataclass
class TrainingArgs:
    base_model_id: str = "bert-base-uncased"
    logging_steps: int = 564
    save_steps: int = 1880

    batch_size_per_device: int = 16
    max_epochs: int = 5

    seed: int = 42
    val_split: float = 0.005
    max_length: int = 256

    apply_data_augment: bool = False

    # tx_args
    lr: float = 1e-5
    init_lr: float = 1e-5
    warmup_steps: int = 5640
    weight_decay: float = 0.001

    base_dir: str = "training-expt"
    save_dir: str = "checkpoints"

    data_files: str = "../dataset/train-v2.csv"

    def __post_init__(self):
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_dir = os.path.join(self.base_dir, self.save_dir)
        self.batch_size = self.batch_size_per_device * jax.device_count()


def main(args, logger):
    data = load_dataset("csv", data_files=[args.data_files], split="train")

    browse_node_vocab = build_or_load_vocab(data, column_name="BROWSE_NODE_ID")
    brand_vocab = build_or_load_vocab(data, column_name="BRAND")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)

    data = data.map(lambda x: {"BRAND": IGNORE_IDX if x["BRAND"] is None else brand_vocab[x["BRAND"]]})
    data = data.map(lambda x: {"BROWSE_NODE_ID": browse_node_vocab[str(x["BROWSE_NODE_ID"])]})
    data = preprocess(data, tokenizer.sep_token)

    data = data.map(lambda x: {"len_inputs": len(x["inputs"]) // 4})
    print("data stats:", {
        "max": np.max(data["len_inputs"]),
        "mean": np.mean(data["len_inputs"]),
        "min": np.min(data["len_inputs"]),
    })

    data = data.train_test_split(args.val_split, seed=args.seed)
    print(data)

    data_collator = DataCollator(tokenizer=tokenizer, max_length=args.max_length)
    model = Classifier.from_pretrained(args.base_model_id, num_browse_nodes=len(browse_node_vocab))

    num_train_steps = (len(data["train"]) // args.batch_size) * args.max_epochs
    tx, scheduler = build_tx(
        args.lr, args.init_lr, args.warmup_steps, num_train_steps, args.weight_decay
    )

    trainer = Trainer(
        args=args,
        data_collator=data_collator,
        batchify=batchify,
        train_step_fn=train_step,
        val_step_fn=val_step,
        loss_fn=partial(cls_loss_fn, ignore_idx=IGNORE_IDX),
        model_save_fn=model.save_pretrained,
        logger=logger,
        scheduler_fn=scheduler,
    )

    state = trainer.create_state(model, tx, num_train_steps, ckpt_dir=None)

    try:
        trainer.train(state, data["train"], data["test"], apply_data_augment=args.apply_data_augment)
    except KeyboardInterrupt:
        print("Interrupting training from KEYBOARD")

    print("saving final model")
    model.save_pretrained("final-weights")


if __name__ == "__main__":
    args = TrainingArgs()

    print("##########################")
    print("DEVICES:", jax.devices())
    print("Training with global batch size of", args.batch_size)
    print("##########################")

    logger = wandb.init(project="amazon-ml-hack", config=asdict(args))
    args = replace(args, **dict(wandb.config))
    print(args)

    main(args, logger)
