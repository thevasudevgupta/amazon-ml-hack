from typing import Optional

import flax.linen as nn
import jax
from transformers import FlaxBertModel
from transformers.models.bert.modeling_flax_bert import FlaxBertModule


class ClassifierModule(FlaxBertModule):
    num_browse_nodes: int = None
    num_brands: Optional[int] = None

    def setup(self):
        super().setup()
        self.cls1 = nn.Dense(self.num_browse_nodes, dtype=self.dtype)
        if self.num_brands is not None:
            self.cls2 = nn.Dense(self.num_brands, dtype=self.dtype)

    def __call__(self, *args, **kwargs):
        cls_logits = super().__call__(*args, **kwargs)[1]
        browse_node_logits = self.cls1(cls_logits)
        brand_logits = self.cls2(cls_logits) if self.num_brands is not None else None
        return browse_node_logits, brand_logits


class Classifier(FlaxBertModel):
    module_class = ClassifierModule
