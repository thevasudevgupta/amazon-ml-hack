from transformers import FlaxAutoModel, FlaxPreTrainedModel, PretrainedConfig

from typing import Optional
import flax.linen as nn
import jax.numpy as jnp

class ClassifierConfig(PretrainedConfig):
    base_model_id: str
    num_browse_nodes: int
    num_brands: Optional[int] = None


class ClassifierModule(nn.Module):
    config: ClassifierConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.base_model = FlaxAutoModel.from_pretrained(self.config.model_id, dtype=self.dtype)
        self.pooler = nn.Dense(768, dtype=self.dtype)
        self.cls1 = nn.Dense(self.config.num_browse_nodes, dtype=self.dtype)
        if self.config.num_brands is not None:
            self.cls2 = nn.Dense(self.config.num_brands, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask=None):
        hidden_states = self.base_model(input_ids, attention_mask=attention_mask)
        cls_logits = nn.tanh(self.pooler(hidden_states[:, 0, :]))
        browse_node_logits = self.cls1(cls_logits)
        brand_logits = self.cls2(cls_logits) if self.config.num_brands is not None else None
        return browse_node_logits, brand_logits


class Classifier(FlaxPreTrainedModel):
    module_class = ClassifierModule

    def __init__(
        self,
        config: ClassifierConfig,
        dtype: jnp.dtype = jnp.float32,
        seed: int = 0,
    ):
        module = self.module_class(config=config, dtype=dtype)
        super().__init__(config, module, input_shape=(1,1), seed=seed, dtype=dtype)
