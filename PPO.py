import jax.numpy as jnp, jax
from jax import random
from jax import lax
from jax.tree_util import register_pytree_node_class
from functools import partial
from Sim import Sim, SimCfg, SimData
import optax
from flax.training import checkpoints
from pathlib import Path
from normlize import meanvar_init, meanvar_normalize, meanvar_update, MeanVarState

# qpos is [ballxyz, ballquat, paddelxyz]
class PPO:
    def __init__(
        self,
        policy_model,
        value_model,
        Sim,
        key=random.PRNGKey(0),
        num_envs: int       = 512,
        unroll_length: int  = 32,
        minibatch_size: int = 256,
        ppo_epochs: int     = 4,
        lr: float           = 3e-4,
        gamma: float        = 0.97,
        lam: float          = 0.95,
    ):
        self.policy_model = policy_model
        self.value_model = value_model
        self.Sim = Sim
        self.key = key
        self.gamma = gamma
        self.llambda = lam

        self.unroll_length = unroll_length
        self.epochs = 100
        self.minibatch_size = minibatch_size
        self.ppo_epochs = ppo_epochs

        opt_lr = lr
        self.policy_opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=opt_lr)
        )
        self.value_opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=opt_lr)
        )
        self.policy_opt_state = self.policy_opt.init(self.policy_model)
        self.value_opt_state = self.value_opt.init(self.value_model)

        
    @staticmethod
    @jax.jit
    def reward(prevState: SimData, nextState: SimData, ctrls: jnp.ndarray, prev_ctrls: jnp.ndarray):
        qpos = nextState.qpos
        qvel = nextState.qvel


        prev_qpos = prevState.qpos
        prev_qvel = prevState.qvel

        step = nextState.step

        _SCALE = dict(
            dist        =  3.0,      # shaping: closer paddle‑ball distance
            approach    =  2.0,      # reward getting closer each frame
            hit_bonus   = 10.0,      # extra for contact, scaled by paddle vx
            target      = 15.0,      # ball crosses net near centre‑line
            dir         = 10.0,      # ball velocity mostly +x
            ctrl_cost   = -0.02,     # regularise energy
            miss        = -25.0,     # ball dropped / paddle too low
            end         = 50.0,      # ball leaves table heading forward
        )

        @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0))
        def _reward_and_done(qpos      : jnp.ndarray,
                            qvel      : jnp.ndarray,
                            prev_qpos : jnp.ndarray,
                            prev_qvel : jnp.ndarray,
                            ctrl      : jnp.ndarray,
                            prev_ctrl : jnp.ndarray,
                            step : jnp.ndarray):
            # ---- constants -----------------------------------------------------------
            TABLE_X_LIMIT  = -1.4           # ball past player → miss
            FLOOR_Z_LIMIT  =  0.7           # ball hits floor  → miss
            PADDLE_Z_LIMIT =  0.9           # paddle dropped   → miss
            CONTACT_THRESH =  0.06          # paddle‑ball distance considered a hit

            # ---- unpack state --------------------------------------------------------
            ball_pos        = qpos[:3]
            ball_vel        = qvel[:3]
            prev_ball_pos   = prev_qpos[:3]

            paddle_pos      = qpos[7:10]
            paddle_vel      = qvel[7:10]
            prev_paddle_pos = prev_qpos[7:10]

            # ---- shaping terms -------------------------------------------------------
            d      = jnp.linalg.norm(paddle_pos - ball_pos)          # current distance
            prev_d = jnp.linalg.norm(prev_paddle_pos - prev_ball_pos)


            # ---- ballistic target term ----------------------------------------------
            # Predict lateral offset 0.1 s ahead and encourage centre hits (y≈0)
            t_pred     = 0.1
            future_y   = ball_pos[1] + ball_vel[1] * t_pred
            target_err = jnp.abs(future_y)
            hit_target = 1.0 - jnp.tanh(2.5 * target_err)            # ∈ (0,1]

            # ---- down‑table velocity term -------------------------------------------
            target_dir  = jnp.array([1.0, 0.0, 0.0])
            speed       = jnp.linalg.norm(ball_vel) + 1e-6
            forward_cos = jnp.dot(ball_vel, target_dir) / speed
            dir_reward  = jnp.maximum(0.0, forward_cos) * (speed * 0.5)

            # ---- contact / termination ----------------------------------------------
            reward_dist = jnp.exp(-2.0 * d)           # softer fall‑off
            reward_approach = (prev_d - d) * 30.0     # raw cm change *big* weight

            hit_bonus = (d < 0.08).astype(jnp.float32) * 10.0 * jnp.maximum(0, paddle_vel[0])

            miss = (
                (ball_pos[0] < TABLE_X_LIMIT) |
                (ball_pos[2] < FLOOR_Z_LIMIT) |
                (paddle_pos[2] < PADDLE_Z_LIMIT)
            )
            end  = ball_vel[0] > 0.5                                 # ball gone forward
            done = miss | end

            timeout = step >= 612
            done = done | timeout

            # ---- costs ---------------------------------------------------------------
            ctrl_cost = -jnp.sum(ctrl ** 2)

            # ---- final reward --------------------------------------------------------
            reward = (
                _SCALE["dist"]     * reward_dist +
                _SCALE["approach"] * reward_approach +
                _SCALE["hit_bonus"]* hit_bonus +
                _SCALE["target"]   * hit_target +
                _SCALE["dir"]      * dir_reward +
                _SCALE["ctrl_cost"]* ctrl_cost +
                _SCALE["miss"]     * miss.astype(jnp.float32) +
                _SCALE["end"]      * end.astype(jnp.float32)
            )

            # ---- numerical‑safety guard ---------------------------------------------
            invalid_state = jnp.isnan(ball_pos).any() | jnp.isinf(ball_pos).any()
            reward = jnp.where(invalid_state, -25.0, reward)

            return reward, done



        # Call the vmapped function on the batch of states
        return _reward_and_done(qpos, qvel, prev_qpos, prev_qvel, ctrls, prev_ctrls, step)

    @partial(jax.jit, static_argnums=(0,))
    def rollout(self, policy_Model, value_Model, key, state, obs_state):

        

        def _rollout(carry, _):
            state, key, prev_ctrl, obs_state = carry

            s = self.Sim.getObs(state)   # (B, 19)   

            obs_state = meanvar_update(obs_state, s)
            s = meanvar_normalize(obs_state, s)

            
            x_in = s
            key, subkey = random.split(key)
            subkeys = random.split(subkey, self.Sim.cfg.batch)
            ctrl, std, mu  = policy_Model(x_in, subkeys)
            log_p = policy_Model.calc_log_prob(ctrl, mu, std) 

            V_theta_ts = value_Model(x_in).squeeze(-1)                               

            next_state  = self.Sim.step(state, ctrl)     # (B, SimData), (B, 3)
            r, done = self.reward(state, next_state, ctrl, prev_ctrl)  # (B,), (B,)


            next_state = self.Sim.reset_partial(next_state, done)

            key, subkey = random.split(key)
            noise_qpos = random.normal(subkey, (self.Sim.cfg.batch, 3)) * .1
            noise_qpos = jnp.zeros_like(next_state.qpos).at[:, 7:].set(noise_qpos) * done[:, None]
            key, subkey = random.split(key)
            noise_qvel = random.normal(subkey, (self.Sim.cfg.batch, 3))
            noise_qvel = jnp.zeros_like(next_state.qvel).at[:, :3].set(noise_qvel) * done[:, None]
            next_state = SimData(noise_qpos + next_state.qpos, noise_qvel + next_state.qvel, next_state.step)

            prev_ctrl_next = jnp.where(done[:, None], jnp.zeros_like(ctrl), ctrl)

            return (next_state, key, prev_ctrl_next, obs_state), (log_p, s, r, V_theta_ts, ctrl, done)
        
        prev_ctrl = jnp.zeros((state.qpos.shape[0], 3))
        
        (state, key, _, obs_state), (log_probs_old, states, rewards, V_theta_ts, ctrls, dones) = jax.lax.scan(_rollout, (state, key, prev_ctrl, obs_state), None, length=self.unroll_length)
        # log_probs_old (T, B)
        # states (T, B, 19)
        # rewards (T, B)
        # V_theta_ts (T, B)
        # ctrls (T, B, 7)


        def _compute_A_t_and_V_s_t(carry, xs):
            V_theta_t_plus_1, A_t_plus_1 = carry
            V_theta_t, r_t, done = xs
            mask = done.astype(V_theta_t_plus_1.dtype)

            delta = r_t + (1 - mask) * self.gamma * V_theta_t_plus_1 - V_theta_t

            A_t = delta + self.gamma * self.llambda * (1 - mask) * A_t_plus_1

            V_s_t = V_theta_t + A_t

            return (V_theta_t, A_t), (V_s_t, A_t)
        
        A_t_plus_1 = jnp.zeros(V_theta_ts.shape[-1])
        V_theta_t_plus_1 = V_theta_ts[-1]

        (_, _), (V_s_ts, A_ts) = jax.lax.scan(_compute_A_t_and_V_s_t, (V_theta_t_plus_1, A_t_plus_1), (V_theta_ts, rewards, dones), length=self.unroll_length, reverse=True)
        # A_ts and V_ts are both (T, B)

        T, B = self.unroll_length, self.Sim.cfg.batch
        
        flat_logp     = jax.lax.stop_gradient(log_probs_old.reshape((T * B,)))
        flat_states   = jax.lax.stop_gradient(states.reshape((T * B, -1)))
        flat_rewards  = jax.lax.stop_gradient(rewards.reshape((T * B,)))
        flat_V_s_ts    = jax.lax.stop_gradient(V_s_ts.reshape((T * B,)))
        flat_A_ts     = jax.lax.stop_gradient(A_ts.reshape((T * B,)))
        flat_ctrls    = jax.lax.stop_gradient(ctrls.reshape((T * B, -1)))       

        key, subkey = random.split(key)

        perm = jax.random.permutation(subkey, T * B)

        flat_logp = flat_logp[perm]
        flat_states = flat_states[perm]
        flat_rewards = flat_rewards[perm]
        flat_V_s_ts = flat_V_s_ts[perm]
        flat_A_ts = flat_A_ts[perm]
        flat_ctrls = flat_ctrls[perm]

        Batch = {
            "states": flat_states,
            "logp_old": flat_logp,
            "rewards": flat_rewards,
            "V_targets": flat_V_s_ts,
            "advantages": flat_A_ts,
            "old_ctrls": flat_ctrls
        }

        return Batch, key, state, obs_state
    
    @staticmethod
    @jax.jit
    def loss_fn_policy(policy_Model, A_ts, old_log_p, states, old_ctrls, key): #Work in batches

        x_in = states
        _, stds, mus  = policy_Model(x_in, key)
        new_log_p = policy_Model.calc_log_prob(old_ctrls, mus, stds)

        A_ts = (A_ts - A_ts.mean()) / (A_ts.std() + 1e-6)

        A_ts = jnp.clip(A_ts, -5.0, 5.0)



        log_ratio = new_log_p - old_log_p
        ratio     = jnp.exp(jnp.clip(log_ratio, -10.0, 10.0))   # pre‑exp clip
        eps       = 0.2

        ratio_clipped = jnp.clip(ratio, 1 - eps, 1 + eps)
        L_P       = -jnp.mean(jnp.minimum(ratio * A_ts, ratio_clipped * A_ts))

        L_P = jnp.nan_to_num(L_P, nan=0.0, posinf=1e3, neginf=-1e3)

        log_stds = jnp.log(stds + 1e-6)
        L_E = jnp.mean(jnp.sum(0.5 * (1 + jnp.log(2*jnp.pi) + 2*log_stds), axis=-1))

        kl   = jnp.mean(old_log_p - new_log_p)

        kl = jnp.mean(jnp.clip(old_log_p - new_log_p, -500.0, 500.0))

        L_E = jnp.clip(L_E, -10.0, 10.0)
        loss = L_P - 0.01 * L_E + 0.01 * kl
        loss = jnp.nan_to_num(loss, nan=0.0, posinf=1e3, neginf=-1e3)
        return loss
    
    @staticmethod
    @jax.jit
    def loss_fn_value(value_Model, V_ts, states, old_ctrls): #Work in batches
        
        clip_v = .1
        x_in = states
        V_pred = value_Model(states).squeeze(-1)          # shape (B,)

        # clipped target: old prediction ± clip_v
        V_target_clipped = V_pred + jnp.clip(V_ts - V_pred,
                                            -clip_v, clip_v)

        # two MSE losses: unclipped vs. clipped target
        mse_unclipped = (V_pred - V_ts)          ** 2
        mse_clipped   = (V_pred - V_target_clipped)  ** 2

        # final loss: 0.5 * mean(max(mse_unclipped, mse_clipped))
        L_V = 0.5 * jnp.mean(jnp.maximum(mse_unclipped, mse_clipped))

        return L_V
    
    
    def train(self):

        policy_loss_grad_fn = jax.jit(jax.value_and_grad(self.loss_fn_policy))
        value_loss_grad_fn = jax.jit(jax.value_and_grad(self.loss_fn_value))

        state = self.Sim.reset() # (B, SimData)
        obs_state  = meanvar_init((self.policy_model.size_in,))

        for epoch in range(self.epochs):

            batch, self.key, state, obs_state = self.rollout(self.policy_model, self.value_model, self.key, state, obs_state)

            # Print metrics from the collected data
            mean_reward = batch['rewards'].mean().item()
            print(f"Epoch {epoch:3d} | mean reward/step : {mean_reward:8.4f}")
            mean_v_targe = batch['V_targets'].mean().item()
            print(f"Epoch {epoch:3d} | mean v_targe/step : {mean_v_targe:8.4f}")

            nan_state  = jnp.isnan(batch['states']).any()
            nan_reward = jnp.isnan(batch['rewards']).any()
            nan_value  = jnp.isnan(batch['V_targets']).any()
            print(f"[epoch {epoch}] nan_state={nan_state}  nan_reward={nan_reward}  nan_value={nan_value}")

            v_pred_1  = jnp.percentile(batch['V_targets'], 1).item()
            v_pred_99 = jnp.percentile(batch['V_targets'], 99).item()
            print(f"Epoch {epoch:3d} | V_target 1‑99%  : {v_pred_1:6.1f} … {v_pred_99:6.1f}")


            def _run_epoch(carry, xs):
                policy_model, value_model, policy_opt_state, value_opt_state, batch, key = carry
                N = batch["logp_old"].shape[0]

                key, subkey = random.split(key)
                perm = jax.random.permutation(subkey, N)
                batch = jax.tree_util.tree_map(lambda x: x[perm], batch)

                def _run_mini_batch_epoch(carry, xs):
                    policy_model, value_model, policy_opt_state, value_opt_state, batch, key = carry
                    j = xs
                    mini_batch = jax.tree_util.tree_map(lambda x: lax.dynamic_slice_in_dim(x, start_index=j, slice_size=self.minibatch_size, axis=0), batch)

                    key, subkey = random.split(key)
                    subkeys = random.split(subkey, self.minibatch_size)
                    

                    policy_loss, policy_grads = policy_loss_grad_fn(policy_model, mini_batch["advantages"], mini_batch["logp_old"], mini_batch["states"], mini_batch["old_ctrls"], subkeys)
                    
                    policy_updates, policy_opt_state = self.policy_opt.update(policy_grads, policy_opt_state, policy_model)
                    policy_model = optax.apply_updates(policy_model, policy_updates)

                    value_loss, value_grads = value_loss_grad_fn(value_model, mini_batch["V_targets"], mini_batch["states"], mini_batch["old_ctrls"])
                    
                    value_updates, value_opt_state = self.value_opt.update(value_grads, value_opt_state, value_model)
                    value_model = optax.apply_updates(value_model, value_updates)
                    
                    return (policy_model, value_model, policy_opt_state, value_opt_state, batch, key), (value_loss + policy_loss)
                
                mini_epoch_starting = jnp.arange(0, N, self.minibatch_size)

                (policy_model, value_model, policy_opt_state, value_opt_state, _, key), losses = jax.lax.scan(_run_mini_batch_epoch, (policy_model, value_model, policy_opt_state, value_opt_state, batch, key), mini_epoch_starting, length=mini_epoch_starting.shape[0])

                return (policy_model, value_model, policy_opt_state, value_opt_state, batch, key), jnp.mean(losses)
            
            (self.policy_model, self.value_model, self.policy_opt_state, self.value_opt_state, _, self.key), losses = jax.lax.scan(_run_epoch, (self.policy_model, self.value_model, self.policy_opt_state, self.value_opt_state, batch, self.key), None, length=self.ppo_epochs)
            
            # Print the mean loss from the *last* PPO epoch as a representative value
            print(f"          | mean loss: {losses[-1]:8.4f}")

            # ---- Checkpoint Saving ----
            ckpt_dir = Path("checkpoints").resolve()
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            checkpoints.save_checkpoint(
                ckpt_dir=ckpt_dir,
                target=self.policy_model,
                step=epoch,
                prefix="policy_",
                overwrite=True
            )
            # print(f"          | ✓ saved weights to {ckpt_dir}/policy_{epoch}")