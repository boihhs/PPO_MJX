import jax.numpy as jnp, jax
from jax import random
from jax.tree_util import register_pytree_node_class
from functools import partial


@register_pytree_node_class
class Value:
    def __init__(self, size_in, size_out, key=random.PRNGKey(0)):
        self.size_in, self.size_out = size_in, size_out
        self.key = key
        # three dense layers as before
        self.w1, self.b1, self.key = self._layer(size_in, 256, self.key, bias_scale=0.0, use_normal=True)
        self.w2, self.b2, self.key = self._layer(256,     256, self.key, bias_scale=0.0, use_normal=True)
        self.w3, self.b3, self.key = self._layer(256,     256, self.key, bias_scale=0.0, use_normal=True)
        self.w4, self.b4, self.key = self._layer(256,     256, self.key, bias_scale=0.0, use_normal=True)
        self.w5, self.b5, self.key = self._layer(256,     size_out, self.key, bias_scale=0.0, use_normal=True)


    @staticmethod
    def _layer(m, n, key, *, bias_scale=0.0, use_normal=True):
        key, sub_w, sub_b = random.split(key, 3)

        if use_normal:
            std = jnp.sqrt(2.0 / (m + n))
            w = std * random.normal(sub_w, (n, m))
        else:
            limit = jnp.sqrt(6.0 / (m + n))
            w = random.uniform(sub_w, (n, m), minval=-limit, maxval=limit)

        b = jnp.zeros((n,))
        return w, b, key

    # ── forward ────────────────────────────────────────────────────────────
    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def __call__(self, x):
        x = jnp.clip(x, -1e3, 1e3)

        x = jax.nn.swish(self.w1 @ x + self.b1)
        x = jax.nn.swish(self.w2 @ x + self.b2)
        x = jax.nn.swish(self.w3 @ x + self.b3)
        x = jax.nn.swish(self.w4 @ x + self.b4)
        raw = self.w5 @ x + self.b5 # raw output of shape (7,)

        return raw

    # ── pytree interface (unchanged) ───────────────────────────────────────
    def tree_flatten(self):
        children = (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4, self.w5, self.b5)
        aux = (self.size_in, self.size_out)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        size_in, size_out = aux
        obj = cls.__new__(cls)  # bypass __init__
        (obj.w1, obj.b1, obj.w2, obj.b2, obj.w3, obj.b3, obj.w4, obj.b4, obj.w5, obj.b5) = children
        obj.size_in, obj.size_out = size_in, size_out
        obj.key = None
        return obj
