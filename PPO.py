import jax.numpy as jnp, jax
from jax import random
from jax import lax
from jax.tree_util import register_pytree_node_class
from functools import partial
from Sim import Sim, SimCfg, SimData
import optax
from flax.training import checkpoints
from pathlib import Path

# qpos is [ballxyz, ballquat, paddelxyz]
class PPO:
    def __init__(self, policy_model, value_model, Sim, key=random.PRNGKey(0)):
        self.policy_model = policy_model
        self.value_model = value_model
        self.Sim = Sim
        self.key = key
        self.gamma = .9
        self.llambda = .9

        # In your PPO __init__
        initial_lr = 3e-5

        self.steps = 3000

        self.epochs=100
        self.minibatch_size=256
        self.ppo_epochs=4

        
        self.policy_opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=3e-4)
        )
        self.value_opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=1e-4)
        )
        self.policy_opt_state = self.policy_opt.init(self.policy_model)
        self.value_opt_state = self.value_opt.init(self.value_model)
        
    @staticmethod
    @jax.jit
    def reward(prevState: SimData, nextState: SimData, ctrls: jnp.ndarray, contacts: jnp.ndarray, prev_ctrls: jnp.ndarray):
        qpos = nextState.qpos
        qvel = nextState.qvel

        prev_qpos = prevState.qpos
        prev_qvel = prevState.qvel

        @partial(jax.vmap, in_axes=(0,0,0,0,0,0, 0), out_axes=(0,0))
        def _reward_and_done(qpos: jnp.ndarray, qvel: jnp.ndarray, prev_qpos: jnp.ndarray, prev_qvel: jnp.ndarray, ctrl: jnp.ndarray, contact: jnp.ndarray, prev_ctrl: jnp.ndarray):

            PADDLE_CENTER = jnp.array([-1.5, 0.0, 1.0])
            SPHERE_RADIUS = 2.0
            k_sphere      = 0.05

            paddle_ball_contact  = contact[0]
            table_ball_contact   = contact[1]
            ball_net_contact     = contact[2]
            paddle_table_contact = contact[3]

            ball_pos        = qpos[:3]
            ball_vel = qvel[:3]
            prev_ball_pos   = prev_qpos[:3]
            paddle_pos      = qpos[7:10]
            prev_paddle_pos = prev_qpos[7:10]
            paddle_vel      = qvel[6:9]
            prev_paddle_vel = prev_qvel[6:9]

            d      = jnp.linalg.norm(paddle_pos - ball_pos)          # current dist
            prev_d = jnp.linalg.norm(prev_paddle_pos - prev_ball_pos)

            reward_dist  = jnp.exp(-3.0 * d)                         # in (0,1]
            reward_prog  = (d < prev_d).astype(jnp.float32)          # 1 if closer
            # (we’ll zero this out if episode just ended, see below)

            # ----------------- directional reward -----------------
            # Want ball moving fast along +x and *not* along y or z
            target_dir = jnp.array([1.0, 0.0, 0.0])
            speed      = jnp.linalg.norm(ball_vel) + 1e-6
            forward_cos = jnp.dot(ball_vel, target_dir) / speed      # ∈[-1,1]

            # reward >0 only if ball is moving forward
            dir_reward = jnp.maximum(0.0, forward_cos) * speed       # 0 if back/side
            dir_reward = jnp.clip(dir_reward, 0.0, 20.0) * 10

            # penalise off‑axis motion (y, z components)
          


            # ----------------- contact & termination --------------
            hit   = d < 0.06                                        # contact event
            hit_bonus = 10.0 * hit * jnp.maximum(0.0, paddle_vel[0]) # harder = better

            

            miss = (ball_pos[0] < -1.4) | (ball_pos[2] < 0.7) | (paddle_pos[2] < 0.9)
            end  = ball_vel[0] > .5                                # ball clearly gone

            invalid_state = jnp.isnan(ball_pos).any() | jnp.isinf(ball_pos).any()
            miss = miss | invalid_state
            done = miss | end

            # ----------------- costs ------------------------------
            ctrl_cost = -1e-2 * jnp.sum(ctrl ** 2)

            # ----------------- episode‑boundary fix ---------------
            # no progress reward immediately *after* a reset
            reward_prog = jnp.where(done, 0.0, reward_prog)

            # ----------------- final reward -----------------------
            reward = (
                3.0 * reward_dist                 # smaller shaping weight
                + reward_prog * 3.0
                + hit_bonus
                + dir_reward      # much stronger signal early on
                + ctrl_cost
                + miss * -25.0
                + end  * 50.0
                )
            

            return reward, done



        # Call the vmapped function on the batch of states
        return _reward_and_done(qpos, qvel, prev_qpos, prev_qvel, ctrls, contacts, prev_ctrls)

    @partial(jax.jit, static_argnums=(0,))
    def rollout(self, policy_Model, value_Model, key):

        state = self.Sim.reset() # (B, SimData)

        noise_qpos = jnp.zeros(state.qpos.shape)
        noise_qvel = jnp.zeros(state.qvel.shape)

        key, subkey = random.split(key)

        noise = random.normal(subkey, (self.Sim.cfg.batch, 3)) * .1

        key, subkey = random.split(key)

        noise_vel = random.normal(subkey, (self.Sim.cfg.batch, 3))

        noise_qpos = noise_qpos.at[:, 7:].set(noise)
        noise_qvel = noise_qvel.at[:, :3].set(noise_vel)

        state = SimData(noise_qpos + state.qpos, noise_qvel + state.qvel) # (B, SimData)

        def _rollout(carry, _):
            state, key, prev_ctrl = carry

            s = self.Sim.getObs(state)   # (B, 19)       
            
            x_in = s
            key, subkey = random.split(key)
            subkeys = random.split(subkey, self.Sim.cfg.batch)
            ctrl, std, mu  = policy_Model(x_in, subkeys)
            log_p = policy_Model.calc_log_prob(ctrl, mu, std) 

            V_theta_ts = value_Model(x_in).squeeze(-1)                               

            next_state, contacts = self.Sim.step(state, ctrl)     # (B, SimData), (B, 3)
            r, done = self.reward(state, next_state, ctrl, contacts, prev_ctrl)  # (B,), (B,)


            next_state = self.Sim.reset_partial(next_state, done)

            key, subkey = random.split(key)
            noise_qpos = random.normal(subkey, (self.Sim.cfg.batch, 3)) * .1
            noise_qpos = jnp.zeros_like(next_state.qpos).at[:, 7:].set(noise_qpos) * done[:, None]
            key, subkey = random.split(key)
            noise_qvel = random.normal(subkey, (self.Sim.cfg.batch, 3))
            noise_qvel = jnp.zeros_like(next_state.qvel).at[:, :3].set(noise_qvel) * done[:, None]
            next_state = SimData(noise_qpos + next_state.qpos, noise_qvel + next_state.qvel)

            return (next_state, key, ctrl), (log_p, s, r, V_theta_ts, ctrl, done)
        
        prev_ctrl = jnp.zeros((state.qpos.shape[0], 3))
        
        (state, key, _), (log_probs_old, states, rewards, V_theta_ts, ctrls, dones) = jax.lax.scan(_rollout, (state, key, prev_ctrl), None, length=self.steps)
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
        V_theta_t_plus_1 = value_Model(self.Sim.getObs(state)).squeeze(-1)

        (_, _), (V_s_ts, A_ts) = jax.lax.scan(_compute_A_t_and_V_s_t, (V_theta_t_plus_1, A_t_plus_1), (V_theta_ts, rewards, dones), length=self.steps, reverse=True)
        # A_ts and V_ts are both (T, B)

        T, B = self.steps, self.Sim.cfg.batch
        
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

        return Batch, key
    
    @staticmethod
    @jax.jit
    def loss_fn_policy(policy_Model, A_ts, old_log_p, states, old_ctrls, key): #Work in batches

        x_in = states
        _, stds, mus  = policy_Model(x_in, key)
        new_log_p = policy_Model.calc_log_prob(old_ctrls, mus, stds)

        A_ts = (A_ts - A_ts.mean()) / (A_ts.std() + 1e-6)

        A_ts = jnp.clip(A_ts, -5.0, 5.0)



        log_ratio = new_log_p - old_log_p
        ratio     = jnp.exp(log_ratio)

        eps       = 0.3
        ratio_clipped = jnp.clip(ratio, 1 - eps, 1 + eps)
        L_P       = -jnp.mean(jnp.minimum(ratio * A_ts, ratio_clipped * A_ts))

        L_P = jnp.nan_to_num(L_P, nan=0.0, posinf=1e3, neginf=-1e3)

        L_E = jnp.mean(jnp.sum( 0.5*(jnp.log(2*jnp.pi*stds**2 + 1e-6) + 1), axis=-1))

        kl   = jnp.mean(old_log_p - new_log_p)

        kl = jnp.mean(jnp.clip(old_log_p - new_log_p, -500.0, 500.0))

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

        for epoch in range(self.epochs):

            batch, self.key = self.rollout(self.policy_model, self.value_model, self.key)

            # Print metrics from the collected data
            mean_reward = batch['rewards'].mean().item()
            print(batch["logp_old"].shape[0])
            print(f"Epoch {epoch:3d} | mean reward/step : {mean_reward:8.4f}")
            mean_v_targe = batch['V_targets'].mean().item()
            print(f"Epoch {epoch:3d} | mean v_targe/step : {mean_v_targe:8.4f}")

            nan_state  = jnp.isnan(batch['states']).any()
            nan_reward = jnp.isnan(batch['rewards']).any()
            nan_value  = jnp.isnan(batch['V_targets']).any()
            print(f"[epoch {epoch}] nan_state={nan_state}  nan_reward={nan_reward}  nan_value={nan_value}")


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
                
                mini_epoch_starting = jnp.arange(0, N - self.minibatch_size, self.minibatch_size)

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
            print(f"          | ✓ saved weights to {ckpt_dir}/policy_{epoch}")