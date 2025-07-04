from Sim import Sim, SimCfg, SimData
import mujoco
from mujoco import mjx
from jax import lax
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from jax.tree_util import register_pytree_node_class
import jax.tree_util
from policy import Policy
from value import Value
from jax import random
from PPO import PPO

policy = Policy(19, 6)
value = Value(19, 1)
cfg = SimCfg(
        xml_path="/home/leo-benaharon/Desktop/ping_pong/env_ping_pong.xml",
        batch    = 128,
        init_pos = jnp.array([3.5, 0, 1.3, 1, 0, 0, 0,   -1.5, 0.0, 1.],
                             dtype=jnp.float32),
        init_vel = jnp.array([-10, 0, 0, 0, 0, 0, 0, 0, 0],
                             dtype=jnp.float32),
    )
ctrl = jnp.zeros((cfg.batch, 3))
sim = Sim(cfg)


ppo = PPO(policy, value, sim)

ppo.train()



