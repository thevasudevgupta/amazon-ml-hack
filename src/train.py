import json
import os
from dataclasses import dataclass
from functools import partial

import wandb
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from data_utils import DataCollator, batchify
from modeling_utils import Classifier
from training_utils import (
    Trainer,
    build_tx,
    cls_loss_fn,
    scheduler_fn,
    train_step,
    val_step,
)


@dataclass
class TrainingArgs:
    base_model_id: str = "bert-base-uncased"
    logging_steps: int = 3000
    save_steps: int = 10500

    batch_size_per_device: int = 1
    max_epochs: int = 5
    seed: int = 42
    val_split: float = 0.05

    # tx_args
    lr: float = 3e-5
    init_lr: float = 0.0
    warmup_steps: int = 20000
    weight_decay: float = 0.0095

    base_dir: str = "training-expt"
    save_dir: str = "checkpoints"

    data_files: str = "dataset/train-v2.csv"

    def __post_init__(self):
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_dir = os.path.join(self.base_dir, self.save_dir)
        self.batch_size = self.batch_size_per_device * jax.device_count()


def build_or_load_vocab(dataset, column_name="BROWSE_NODE_ID"):
    if f"{column_name}.json" not in os.listdir("assets"):
        print(f"building vocab from dataset[{column_name}]", end=" ... ")
        ids = set()
        for sample in tqdm(dataset):
            ids.update(sample[column_name])
        vocab = {id: i for i, id in enumerate(ids)}
        with open(f"assets/{column_name}.json", "w") as f:
            json.dump(vocab, f)
        print("done!!")
    else:
        with open(f"assets/{column_name}.json") as f:
            vocab = json.load(f)
    return vocab


def main(args, logger):
    data = load_dataset("csv", data_files=[args.data_files], split="train")
    browse_node_vocab = build_or_load_vocab(data, column_name="BROWSE_NODE_ID")
    brand_vocab = build_or_load_vocab(data, column_name="BRAND")

    # extra samples will be dropped
    tr_data, val_data = data.train_test_split(args.val_split, seed=args.seed)
    print(tr_data, val_data)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    data_collator = DataCollator(tokenizer=tokenizer, max_length=args.max_length)

    model = Classifier(
        base_model_id=args.base_model_id,
        num_browse_nodes=len(browse_node_vocab),
        num_brands=len(brand_vocab),
    )

    trainer = Trainer(
        args=args,
        data_collator=data_collator,
        batchify=partial(batchify, sep_token=tokenizer.sep_token),
        train_step_fn=train_step,
        val_step_fn=val_step,
        loss_fn=cls_loss_fn,
        model_save_fn=model.save_pretrained,
        logger=logger,
        scheduler_fn=scheduler_fn,
    )

    num_train_steps = (len(tr_data) // args.batch_size) * args.max_epochs
    tx = build_tx(
        args.lr, args.init_lr, args.warmup_steps, num_train_steps, args.weight_decay
    )
    state = trainer.create_state(model, tx, num_train_steps, ckpt_dir=None)

    try:
        trainer.train(state, tr_data, val_data)
    except KeyboardInterrupt:
        print("Interrupting training from KEYBOARD")

    print("saving final model")
    model.save_pretrained("final-weights")


if __name__ == "__main__":
    args = TrainingArgs()
    logger = wandb.init()
    main(args, logger)
