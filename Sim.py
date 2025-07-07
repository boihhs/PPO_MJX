import mujoco
from mujoco import mjx
from jax import lax
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
import jax.tree_util
from functools import partial

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SimCfg:
    xml_path: str     = field(metadata={'static': True})
    batch: int     = field(metadata={'static': True})
    model_freq: int     = field(metadata={'static': True})
    init_pos: jnp.ndarray
    init_vel: jnp.ndarray

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ContactIDs:
    paddle_id: int     = field(metadata={'static': True})
    ball_id: int     = field(metadata={'static': True})
    table_id: int     = field(metadata={'static': True})
    net_id: int     = field(metadata={'static': True})

@jax.tree_util.register_dataclass
@dataclass
class SimData:
    qpos: jnp.ndarray # (B, np)
    qvel: jnp.ndarray # (B, nv)
    step: jnp.ndarray # (B, )

@jax.tree_util.register_pytree_node_class
class Sim:
    def __init__(self, cfg: SimCfg):
        self.cfg       = cfg
        mj_model       = mujoco.MjModel.from_xml_path(cfg.xml_path)
        mj_model.opt.iterations = 20
        self.timestep = mj_model.opt.timestep
        mj_model.opt.tolerance  = 1e-8

        paddle_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_face")
        ball_id   = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        table_id   = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        net_id   = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "net_geom")

        self.contact_ids = ContactIDs(paddle_id=paddle_id, ball_id=ball_id, table_id=table_id, net_id=net_id)
        self.mjx_model = mjx.put_model(mj_model)
        self.blank     = mjx.make_data(self.mjx_model)


    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0, 0), out_axes=(0))
    def step(self, simDatas: SimData, ctrl):
        blank = self.blank
        phyics_steps_per_model = int((1/self.cfg.model_freq) / self.timestep)

        d = blank.replace(qpos=simDatas.qpos, qvel=simDatas.qvel, ctrl=ctrl)
        
        def _step(context, _):
            d = context
            d = mjx.step(self.mjx_model, d)           
            return d, None
            
        d, _ = jax.lax.scan(_step, d, None, length=phyics_steps_per_model)
        return SimData(d.qpos, d.qvel, simDatas.step + 1)
  
    @jax.jit
    def reset(self):
        @partial(jax.vmap, in_axes=(0), out_axes=(0))
        def _reset(_):
            return SimData(self.cfg.init_pos, self.cfg.init_vel, jnp.array(0))
    
        seq = jnp.arange(self.cfg.batch)
        return _reset(seq)
    
    @jax.jit
    def reset_partial(self, simDatas : SimData, dones): # Batch of simDatas
        @partial(jax.vmap, in_axes=(0, 0), out_axes=(0))
        def _reset(simData:SimData, done):
            mask = done.astype(simData.qpos.dtype)
            new_qpos = self.cfg.init_pos * mask + simData.qpos * (1 - mask)
            new_qvel = self.cfg.init_vel * mask + simData.qvel * (1 - mask)
            new_step = (simData.step) * (1 - mask)
            return SimData(new_qpos, new_qvel, new_step)
    
        return _reset(simDatas, dones)
    
    @staticmethod
    @jax.jit
    def getObs(simData: SimData):
        return jnp.concatenate([simData.qpos, simData.qvel], -1)


    def tree_flatten(self):
        return (), (self.cfg, self.mjx_model, self.blank, self.contact_ids, self.timestep)

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, mjx_model, blank, contact_ids, timestep = aux
        obj = cls.__new__(cls)
        obj.cfg, obj.mjx_model, obj.blank, obj.contact_ids, obj.timestep = cfg, mjx_model, blank, contact_ids, timestep
        return obj
    




    # def _get_contacts(mjx_data):
    #             ncon = mjx_data.ncon                          

    #             c = mjx_data.contact           
    #             geom1 = c.geom1[:ncon] 
    #             geom2 = c.geom2[:ncon]                        

    #             mask_pb = (geom1 == self.contact_ids.paddle_id) & (geom2 == self.contact_ids.ball_id) | (geom1 == self.contact_ids.ball_id)   & (geom2 == self.contact_ids.paddle_id)

    #             mask_tb = (geom1 == self.contact_ids.table_id)  & (geom2 == self.contact_ids.ball_id) | (geom1 == self.contact_ids.ball_id)   & (geom2 == self.contact_ids.table_id)

    #             mask_nb = (geom1 == self.contact_ids.net_id)    & (geom2 == self.contact_ids.ball_id) | (geom1 == self.contact_ids.ball_id)   & (geom2 == self.contact_ids.net_id)

    #             mask_pt = (geom1 == self.contact_ids.paddle_id)    & (geom2 == self.contact_ids.table_id) | (geom1 == self.contact_ids.table_id)   & (geom2 == self.contact_ids.paddle_id)

    #             hit_pb = jnp.any(mask_pb)      
    #             hit_tb = jnp.any(mask_tb)       
    #             hit_nb = jnp.any(mask_nb)
    #             hit_pt = jnp.any(mask_pt)


    #             return jnp.stack([hit_pb, hit_tb, hit_nb, hit_pt])