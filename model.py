import jax.numpy as jnp, jax
from jax import random
from jax.tree_util import register_pytree_node_class
from functools import partial


@register_pytree_node_class
class MLP:
    def __init__(self, size_in, size_out, key=random.PRNGKey(0)):
        self.size_in, self.size_out = size_in, size_out
        self.key = key
        # three dense layers as before
        self.w1, self.b1, self.key = self._layer(size_in,  128, self.key)
        self.w2, self.b2, self.key = self._layer(128, 128, self.key)
        self.w3, self.b3, self.key = self._layer(128, size_out, self.key)

    @staticmethod
    def _layer(m, n, key, scale=1e-2):
        key, sub = random.split(key)
        w = scale * random.normal(sub, (n, m))
        b = scale * random.normal(key, (n,))
        return w, b, key

    # ── forward ────────────────────────────────────────────────────────────
    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def __call__(self, x):
        x = jnp.maximum(0, self.w1 @ x + self.b1)
        x = jnp.maximum(0, self.w2 @ x + self.b2)
        raw = self.w3 @ x + self.b3 # raw output of shape (7,)

        # --- CORRECTED ACTION SQUASHING ---
        action_scale = 10.0 # From ctrlrange in XML
        mu = action_scale * jnp.tanh(raw[:3])
        # -----------------------------------

        log_std = raw[3:6]

        LOG_STD_MIN = -5.0
        LOG_STD_MAX = -2.3
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

        value = raw[6]

        return jnp.concatenate([mu, log_std, value[jnp.newaxis]], axis=0)

    # ── pytree interface (unchanged) ───────────────────────────────────────
    def tree_flatten(self):
        children = (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)
        aux = (self.size_in, self.size_out)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        size_in, size_out = aux
        obj = cls.__new__(cls)  # bypass __init__
        (obj.w1, obj.b1, obj.w2, obj.b2, obj.w3, obj.b3) = children
        obj.size_in, obj.size_out = size_in, size_out
        obj.key = None
        return obj
