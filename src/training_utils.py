import json
import os
from dataclasses import dataclass
from functools import partial
from typing import Callable

import joblib
import wandb
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils, struct, traverse_util
from flax.serialization import from_bytes, to_bytes
from flax.training import train_state
from flax.training.common_utils import shard


def cross_entropy(logits, labels, ignore_idx=-100):
    """
    Args:
        logits: bsz, vocab_size
        labels: bsz
    """
    indices_to_consider = labels != ignore_idx

    vocab_size = logits.shape[-1]
    labels = (labels[..., None] == jnp.arange(vocab_size)[None]).astype("f4")
    logits = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(labels * logits, axis=-1)
    loss = jnp.take(loss, indices_to_consider)
    return jnp.mean(loss)


def cls_loss_fn(browse_node_logits, browse_nodes, brand_logits=None, brands=None, ignore_idx=-100):
    loss = cross_entropy(browse_node_logits, browse_nodes, ignore_idx=ignore_idx)
    if brand_logits is not None and brands is not None:
        loss = (loss + cross_entropy(brand_logits, brands, ignore_idx=ignore_idx)) / 2
    return loss


@partial(jax.pmap, axis_name="batch")
def train_step(state, drp_rng, model_inputs):
    def loss_fn(params):
        browse_nodes = model_inputs.pop("browse_nodes")
        brands = model_inputs.pop("brands", None)

        browse_node_logits, brand_logits = state.apply_fn(
            **model_inputs, params=params, dropout_rng=drp_rng, train=True
        )
        loss = state.loss_fn(
            browse_node_logits, browse_nodes, brand_logits=brand_logits, brands=brands
        )
        return loss

    drp_rng, new_drp_rng = jax.random.split(drp_rng)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    metrics = jax.lax.pmean({"loss": loss}, axis_name="batch")
    grads = jax.lax.pmean(grads, "batch")

    state = state.apply_gradients(grads=grads)
    return state, metrics, new_drp_rng


@partial(jax.pmap, axis_name="batch")
def val_step(state, **model_inputs):
    browse_nodes = model_inputs.pop("browse_nodes")
    brands = model_inputs.pop("brands", None)

    browse_node_logits, brand_logits = state.apply_fn(
        **model_inputs, params=state.params, train=False
    )
    loss = state.loss_fn(
        browse_node_logits, browse_nodes, brand_logits=brand_logits, brands=brands
    )

    metrics = jax.lax.pmean({"loss": loss}, axis_name="batch")
    return metrics


class TrainState(train_state.TrainState):
    loss_fn: Callable = struct.field(pytree_node=False)


@dataclass
class Trainer:
    args: object
    data_collator: Callable
    batchify: Callable

    train_step_fn: Callable
    val_step_fn: Callable
    loss_fn: Callable

    model_save_fn: Callable
    logger: wandb

    scheduler_fn: Callable = None

    def create_state(self, model, tx, num_train_steps, ckpt_dir=None):
        params = model.params
        state = TrainState.create(
            apply_fn=model.__call__,
            params=params,
            tx=tx,
            loss_fn=self.loss_fn,
        )
        if ckpt_dir is not None:
            params, opt_state, step, args, data_collator = restore_checkpoint(
                ckpt_dir, state
            )
            tx_args = {
                "lr": args.lr,
                "init_lr": args.init_lr,
                "warmup_steps": args.warmup_steps,
                "num_train_steps": num_train_steps,
                "weight_decay": args.weight_decay,
            }
            tx, lr = build_tx(**tx_args)
            state = train_state.TrainState(
                step=step,
                apply_fn=model.__call__,
                params=params,
                tx=tx,
                opt_state=opt_state,
            )
            self.args = args
            self.data_collator = data_collator
            self.scheduler_fn = lr
            model.params = params
        return jax_utils.replicate(state)

    def train(self, state, tr_dataset, val_dataset, apply_data_augment=False):
        args = self.args
        total = len(tr_dataset) // args.batch_size

        rng = jax.random.PRNGKey(0)
        drp_rng = jax.random.split(rng, jax.device_count())
        for epoch in range(args.max_epochs):
            running_loss = jnp.array(0, dtype=jnp.float32)

            if apply_data_augment:
                print(tr_dataset[0]["inputs"])
                tr_dataset = tr_dataset.map(lambda x: {"inputs": get_noisy_sent(x["inputs"].split())}, load_from_cache_file=False)
                print(tr_dataset[0]["inputs"])
            tr_dataloader = self.batchify(tr_dataset, args.batch_size, seed=epoch)
            i = 0
            for batch in tqdm(
                tr_dataloader, total=total, desc=f"Running EPOCH-{epoch}"
            ):
                batch = shard(self.data_collator(batch))
                state, metrics, drp_rng = self.train_step_fn(state, drp_rng, batch)
                running_loss += jax_utils.unreplicate(metrics["loss"])
                i += 1
                if i % args.logging_steps == 0:
                    state_step = jax_utils.unreplicate(state.step)
                    tr_loss = running_loss.item() / i
                    lr = self.scheduler_fn(state_step - 1)

                    eval_loss = self.evaluate(state, val_dataset)
                    logging_dict = dict(
                        step=state_step.item(),
                        eval_loss=eval_loss.item(),
                        tr_loss=tr_loss,
                        lr=lr.item(),
                    )
                    tqdm.write(str(logging_dict))
                    self.logger.log(logging_dict, commit=True)

                if i % args.save_steps == 0:
                    self.save_checkpoint(args.save_dir + f"-e{epoch}-s{i}", state=state)

    def evaluate(self, state, dataset):
        dataloader = self.batchify(dataset, self.args.batch_size)
        total = len(dataset) // self.args.batch_size
        running_loss = jnp.array(0, dtype=jnp.float32)
        i = 0
        for batch in tqdm(dataloader, total=total, desc="Evaluating ... "):
            batch = shard(self.data_collator(batch))
            metrics = self.val_step_fn(state, **batch)
            running_loss += jax_utils.unreplicate(metrics["loss"])
            i += 1
        return running_loss / i

    def save_checkpoint(self, save_dir, state):
        state = jax_utils.unreplicate(state)
        print(f"SAVING CHECKPOINT IN {save_dir}", end=" ... ")
        self.model_save_fn(save_dir, params=state.params)
        with open(os.path.join(save_dir, "opt_state.msgpack"), "wb") as f:
            f.write(to_bytes(state.opt_state))
        joblib.dump(self.args, os.path.join(save_dir, "args.joblib"))
        joblib.dump(self.data_collator, os.path.join(save_dir, "data_collator.joblib"))
        with open(os.path.join(save_dir, "training_state.json"), "w") as f:
            json.dump({"step": state.step.item()}, f)
        print("DONE")


def restore_checkpoint(save_dir, state):
    print(f"RESTORING CHECKPOINT FROM {save_dir}", end=" ... ")
    with open(os.path.join(save_dir, "flax_model.msgpack"), "rb") as f:
        params = from_bytes(state.params, f.read())

    with open(os.path.join(save_dir, "opt_state.msgpack"), "rb") as f:
        opt_state = from_bytes(state.opt_state, f.read())

    args = joblib.load(os.path.join(save_dir, "args.joblib"))
    data_collator = joblib.load(os.path.join(save_dir, "data_collator.joblib"))

    with open(os.path.join(save_dir, "training_state.json"), "r") as f:
        training_state = json.load(f)
    step = training_state["step"]

    print("DONE")
    return params, opt_state, step, args, data_collator


def scheduler_fn(lr, init_lr, warmup_steps, num_train_steps):
    decay_steps = num_train_steps - warmup_steps
    warmup_fn = optax.linear_schedule(
        init_value=init_lr, end_value=lr, transition_steps=warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=lr, end_value=1e-7, transition_steps=decay_steps
    )
    lr = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps]
    )
    return lr


def build_tx(lr, init_lr, warmup_steps, num_train_steps, weight_decay):
    def weight_decay_mask(params):
        params = traverse_util.flatten_dict(params)
        mask = {
            k: (v[-1] != "bias" and v[-2:] != ("LayerNorm", "scale"))
            for k, v in params.items()
        }
        return traverse_util.unflatten_dict(mask)

    lr = scheduler_fn(lr, init_lr, warmup_steps, num_train_steps)

    tx = optax.adamw(
        learning_rate=lr, weight_decay=weight_decay, mask=weight_decay_mask
    )
    return tx, lr
