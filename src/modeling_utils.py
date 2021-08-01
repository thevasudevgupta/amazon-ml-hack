from transformers import FlaxBertModel
from transformers.models.bert.modeling_flax_bert import FlaxBertModule

from typing import Optional

import jax
import flax.linen as nn


class ClassifierModule(FlaxBertModule):
    num_browse_nodes: int = None
    num_brands: Optional[int] = None
    lambd: float = 1.0

    def setup(self):
        super().setup()
        self.cls1 = nn.Dense(self.num_browse_nodes, dtype=self.dtype)
        if self.num_brands is not None:
            self.cls2 = nn.Dense(self.num_brands, dtype=self.dtype)

    def __call__(self, *args, **kwargs):
        cls_logits = super().__call__(*args, **kwargs)[1]

        mix_rng = self.make_rng("dropout")
        if self.lambd < 1:
            cls_logits = self.lambd * cls_logits + (1 - self.lambd) * jax.random.permutation(mix_rng, cls_logits)
        browse_node_logits = self.cls1(cls_logits)
        if self.lambd < 1:
            browse_node_logits = self.lambd * browse_node_logits + (1 - self.lambd) * jax.random.permutation(mix_rng, browse_node_logits)

        brand_logits = self.cls2(cls_logits) if self.num_brands is not None else None
        return browse_node_logits, brand_logits


class Classifier(FlaxBertModel):
    module_class = ClassifierModule
