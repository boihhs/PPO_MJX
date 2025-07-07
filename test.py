"""
Run a trained PPO policy in MuJoCo viewer at 60 FPS.
"""

import time, re, threading
from pathlib import Path

import mujoco
from mujoco import viewer
from pynput import keyboard

import numpy as np
import jax, jax.numpy as jnp
from flax.training import checkpoints
from jax_mujoco_2.policy import Policy   # ← your MLP definition
from normlize import meanvar_init, meanvar_normalize, meanvar_update, MeanVarState


XML_PATH = "env_ping_pong.xml"
CKPT_DIR = Path("jax_mujoco_2").absolute() / "checkpoints"
CKPT_PREFIX = "policy_"            # adjust if you used a different prefix
DT_TARGET = 1.0 / 60.0             # 60 FPS

# ── 0. MuJoCo sizes ────────────────────────────────────────────────────────
mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
nq, nv, nu = mj_model.nq, mj_model.nv, mj_model.nu

policy          = Policy(19, 6)
params_template = policy                      # no .init needed

ckpt_path = checkpoints.latest_checkpoint(CKPT_DIR, prefix=CKPT_PREFIX)
if ckpt_path:
    params = checkpoints.restore_checkpoint(ckpt_path, target=params_template)
    step   = int(re.search(r"_([0-9]+)$", ckpt_path).group(1))
    print(f"✓ loaded step {step} from {ckpt_path}")
else:
    print("[WARN] no checkpoint found; using random weights.")
    params = params_template

@jax.jit
def act_fn(model, obs, key):
    ctrls, stds  = model(obs[None, :], key)     # remove batch dim
    return ctrls[0], stds[0]                   # deterministic mean action

from jax import random

key = random.PRNGKey(0)
# ── 4. Keys for manual overrides (optional) ────────────────────────────────
pressed_keys = set()
def on_press(key):
    try:    pressed_keys.add(key.char)
    except AttributeError: pressed_keys.add(str(key))
def on_release(key):
    try:    pressed_keys.discard(key.char)
    except AttributeError: pressed_keys.discard(str(key))
keyboard.Listener(on_press=on_press, on_release=on_release).start()

# ── 5. Actuator IDs, camera, keyframe reset ────────────────────────────────
aid_x = mj_model.actuator("motor_paddle_x").id
aid_y = mj_model.actuator("motor_paddle_y").id
aid_z = mj_model.actuator("motor_paddle_z").id
cam_id = mj_model.camera("fixed_cam").id
kf_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "serve")

paddle_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_face")
ball_id   = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
table_id   = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
print(ball_id)
print(table_id)

mj_data = mujoco.MjData(mj_model)
mujoco.mj_resetDataKeyframe(mj_model, mj_data, kf_id)

obs_state = meanvar_init((19,))      # 19 = obs dimension
mujoco.mj_resetDataKeyframe(mj_model, mj_data, kf_id)
ctrl = jnp.array([0, 0, 0])

episode_start = time.time()
DT_CONTROL = 1.0 / 100.0  # 100 Hz control interval

with viewer.launch_passive(mj_model, mj_data) as v:
    while v.is_running():
        frame_start = time.time()

        # 1. observation
        qpos = jnp.asarray(np.copy(mj_data.qpos), dtype=jnp.float32)
        qvel = jnp.asarray(np.copy(mj_data.qvel), dtype=jnp.float32)
        raw_obs  = jnp.concatenate([qpos, qvel])      # (19,)

        obs_state = meanvar_update(obs_state, raw_obs)
        obs       = meanvar_normalize(obs_state, raw_obs)

        # 2. policy → control
        key, subkey = random.split(key)
        ctrls, stds = params.inference(obs[None, :])
        ctrls, stds = ctrls[0], stds[0]

        dist_paddle_ball = jnp.exp(-2 * jnp.linalg.norm(jnp.array([0, 0, 2]) - jnp.array(mj_data.qpos[7:])))
        
        ncon = mj_data.ncon
        c    = mj_data.contact
        geom1 = c.geom1[:ncon]
        geom2 = c.geom2[:ncon]
        mask_pb = ((geom1 == paddle_id) & (geom2 == ball_id)) | ((geom1 == ball_id) & (geom2 == paddle_id))
        hit_pb  = jnp.any(mask_pb)
        mask_tb = ((geom1 == table_id) & (geom2 == ball_id)) | ((geom1 == ball_id) & (geom2 == table_id))
        hit_tb  = jnp.any(mask_tb)
        if hit_pb:
            print(hit_pb)

        mj_data.ctrl[:] = np.asarray(ctrls, dtype=np.float64)
        d = jnp.linalg.norm(mj_data.qpos[7:] - mj_data.qpos[:3])
        if d < .06:
            print(d)
        print("control")
        print(mj_data.ctrl)

        # 3. physics stepping until next control tick
        sim_t0 = mj_data.time
        while (mj_data.time - sim_t0) < DT_CONTROL:
            mujoco.mj_step(mj_model, mj_data)

        # render frame
        v.sync()

        # real‑time pacing
        sleep_t = DT_CONTROL - (time.time() - frame_start)
        if sleep_t > 0:
            time.sleep(sleep_t)

        # auto‑reset
        if (time.time() - episode_start) > 5:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, kf_id)
            episode_start = time.time()


