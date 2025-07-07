import jax.numpy as jnp, jax
from jax import random
from jax.tree_util import register_pytree_node_class
from functools import partial


@register_pytree_node_class
class Policy:
    def __init__(self, size_in, size_out, key=random.PRNGKey(0)):
        self.size_in, self.size_out = size_in, size_out
        self.key = key
        # three dense layers as before
        self.w1, self.b1, self.key = self._layer(size_in, 256, self.key, bias_scale=0.0, use_normal=True)
        self.w2, self.b2, self.key = self._layer(256,     256, self.key, bias_scale=0.0, use_normal=True)
        self.w3, self.b3, self.key = self._layer(256,     256, self.key, bias_scale=0.0, use_normal=True)
        self.w4, self.b4, self.key = self._layer(256,     size_out, self.key, bias_scale=0.0, use_normal=True)


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
    @partial(jax.vmap, in_axes=(None, 0, 0), out_axes=(0, 0, 0))
    def __call__(self, x, key):
        x = jax.nn.swish(self.w1 @ x + self.b1)
        x = jax.nn.swish(self.w2 @ x + self.b2)
        x = jax.nn.swish(self.w3 @ x + self.b3)
        raw = self.w4 @ x + self.b4 # raw output of shape (7,)

        mus = raw[:self.size_out//2]
        log_stds = raw[self.size_out//2:]

        CTRL_BOUND = 10
        min_std = 1e-3
        log_stds = jax.nn.softplus(log_stds) + min_std   
        log_stds = jnp.clip(log_stds, -8.0, 2.0)
        stds     = jnp.exp(log_stds)                                   


        key, subkey = random.split(key)
        noise = random.normal(subkey, stds.shape)

        ctrls = CTRL_BOUND * jnp.tanh(mus + noise * stds)

        return ctrls, stds, mus
    
    @staticmethod
    @jax.jit
    @partial(jax.vmap, in_axes=(0, 0, 0), out_axes=(0))
    def calc_log_prob(ctrls, mus, stds):
        CTRL_BOUND = 10
        safe_maha = ((jnp.arctanh(jnp.clip(ctrls / CTRL_BOUND, -0.999999, 0.999999))
              - mus) ** 2) / (stds ** 2 + 1e-6)
        safe_maha  = jnp.clip(safe_maha, 0.0, 1e3)        #  ← NEW  (caps log_prob ≥ −5e2)

        log_prob = jnp.sum(
            -0.5 * (jnp.log(2 * jnp.pi * stds ** 2 + 1e-6) + safe_maha),
            axis=-1
        ) - jnp.sum(jnp.log(jnp.abs(CTRL_BOUND * (1 - (ctrls / CTRL_BOUND) ** 2)) + 1e-6))

        log_prob = jnp.clip(log_prob, -5e2, 5e2)          #  ← NEW  prevents ±Inf
        return log_prob
    
    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=(0, 0))
    def inference(self, x):
        x = jax.nn.swish(self.w1 @ x + self.b1)
        x = jax.nn.swish(self.w2 @ x + self.b2)
        x = jax.nn.swish(self.w3 @ x + self.b3)
        raw = self.w4 @ x + self.b4 # raw output of shape (7,

        mus = raw[:self.size_out//2]
        log_stds = raw[self.size_out//2:]

        CTRL_BOUND = 10
        min_std = 1e-3
        log_stds = jax.nn.softplus(log_stds) + min_std   
        log_stds = jnp.clip(log_stds, -8.0, 2.0)
        stds     = jnp.exp(log_stds)                               

        ctrls = CTRL_BOUND * jnp.tanh(mus)

        return ctrls, stds

    # ── pytree interface (unchanged) ───────────────────────────────────────
    def tree_flatten(self):
        children = (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4)
        aux = (self.size_in, self.size_out)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        size_in, size_out = aux
        obj = cls.__new__(cls)  # bypass __init__
        (obj.w1, obj.b1, obj.w2, obj.b2, obj.w3, obj.b3, obj.w4, obj.b4) = children
        obj.size_in, obj.size_out = size_in, size_out
        obj.key = random.PRNGKey
        return obj
