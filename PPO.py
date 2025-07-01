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
    def __init__(self, MLP, Sim, key=random.PRNGKey(0)):
        self.MLP = MLP
        self.Sim = Sim
        self.key = key
        self.gamma = .99
        self.llambda = .95

        # In your PPO __init__
        initial_lr = 1e-6

        self.steps = 1024

        
        self.opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=initial_lr)
        )
        self.opt_state = self.opt.init(self.MLP)
        
    @staticmethod
    @jax.jit
    def reward(prevState: SimData, nextState: SimData, ctrls: jnp.ndarray, contacts: jnp.ndarray):
        qpos = nextState.qpos
        qvel = nextState.qvel

        prev_qpos = prevState.qpos
        prev_qvel = prevState.qvel

        @partial(jax.vmap, in_axes=(0,0,0,0,0,0), out_axes=(0,0))
        def _reward_and_done(qpos: jnp.ndarray, qvel: jnp.ndarray, prev_qpos: jnp.ndarray, prev_qvel: jnp.ndarray, ctrl: jnp.ndarray, contact: jnp.ndarray):

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

            k_contact     = 10.0
            k_table       = 1.0
            k_dist        = 0.1
            k_vel_ball    = 0.5
            k_vel_paddle  = 0.5
            k_step        = 0.001

            dist   = jnp.linalg.norm(ball_pos - paddle_pos, axis=-1)
            close = dist < .03
            closeer = dist < .02
            
            reward = - dist + close*10 + closeer*20

            
            done   = False

            return reward, done

        # Call the vmapped function on the batch of states
        return _reward_and_done(qpos, qvel, prev_qpos, prev_qvel, ctrls, contacts)

    @partial(jax.jit, static_argnums=(0,))
    def rollout(self, Model, key, percent_done):

        state = self.Sim.reset() # (B, SimData)

        noise_qpos = jnp.zeros(state.qpos.shape)

        key, subkey = random.split(key)

        noise = random.normal(subkey, (self.Sim.cfg.batch, 3))

        noise_qpos = noise_qpos.at[:, 7:].set(noise)

        state = SimData(noise_qpos + state.qpos, state.qvel) # (B, SimData)

        def _rollout(carry, _):
            state, key, prev_ctrl = carry
            key, subkey = random.split(key)

            s = self.Sim.getObs(state) # (B, 19)
            x_in = jnp.concatenate([s, prev_ctrl], -1) # (B, 21)
            out = Model(x_in) # (B, 7)
            mus, log_stds, V_theta_ts = out[:, :3], out[:, 3:6], out[:, 6]

            stds = jnp.exp(log_stds)

            noise = random.normal(subkey, mus.shape)
       
            ctrl = mus + noise * stds # (B, 3)

            logp = jnp.sum(-0.5 * (jnp.log(2 * jnp.pi * stds**2 + 1e-8) + ((ctrl - mus)**2) / (stds**2 + 1e-8)), -1) # (B)

            next_state, contacts = self.Sim.step(state, ctrl) # (B, SimData), # (B, 3)

            r, done = self.reward(state, next_state, ctrl, contacts) #(B)

            next_state = self.Sim.reset_partial(next_state, done)

            noise_qpos = jnp.zeros_like(next_state.qpos)
            key, subkey = random.split(key)
            noise = random.normal(subkey, (self.Sim.cfg.batch, 3))
            noise_qpos = noise_qpos.at[:, 7:].set(noise) * done[:, None] 
            next_state = SimData(noise_qpos + next_state.qpos, next_state.qvel)

            return (next_state, key, ctrl), (logp, s, r, V_theta_ts, ctrl, done)
        
        prev_ctrl = jnp.zeros((state.qpos.shape[0], 3))
        
        (state, key, _), (log_probs_old, states, rewards, V_theta_ts, ctrls, dones) = jax.lax.scan(_rollout, (state, key, prev_ctrl), None, length=self.steps)
        # log_probs_old (T, B)
        # states (T, B, 19)
        # rewards (T, B)
        # V_theta_ts (T, B)
        # ctrls (T, B, 7)

        def _compute_V_t(carry, xs):
            V_t_plus_1 = carry
            r, done = xs
            mask = done.astype(V_t_plus_1.dtype)

            V_t = r + self.gamma * V_t_plus_1 * (1 - mask)
            return V_t, V_t
        
        V_t_plus_1 = jnp.zeros(rewards.shape[-1])

        _, V_ts = jax.lax.scan(_compute_V_t, V_t_plus_1, (rewards, dones), length=self.steps, reverse=True)

        def _compute_A_t(carry, xs):
            V_theta_t_plus_1, A_t_plus_1 = carry
            V_theta_t, r_t, done = xs
            mask = done.astype(V_theta_t_plus_1.dtype)

            A_t = r_t + self.gamma * V_theta_t_plus_1 * (1 - mask) - V_theta_t + self.gamma * self.llambda * A_t_plus_1 * (1 - mask)

            return (V_theta_t, A_t), A_t
        
        A_t_plus_1 = jnp.zeros(V_theta_ts.shape[-1])
        V_theta_t_plus_1 = jnp.zeros(V_theta_ts.shape[-1])

        (_, _), A_ts = jax.lax.scan(_compute_A_t, (V_theta_t_plus_1, A_t_plus_1), (V_theta_ts, rewards, dones), length=self.steps, reverse=True)
        # A_ts and V_ts are both (T, B)

        T, B = self.steps, self.Sim.cfg.batch
        
        flat_logp     = jax.lax.stop_gradient(log_probs_old.reshape((T * B,)))
        flat_states   = jax.lax.stop_gradient(states.reshape((T * B, -1)))
        flat_rewards  = jax.lax.stop_gradient(rewards.reshape((T * B,)))
        flat_V_ts     = jax.lax.stop_gradient(V_ts.reshape((T * B,)))
        flat_A_ts     = jax.lax.stop_gradient(A_ts.reshape((T * B,)))
        flat_ctrls    = jax.lax.stop_gradient(ctrls.reshape((T * B, -1)))

        flat_A_ts = (flat_A_ts - flat_A_ts.mean()) / (flat_A_ts.std() + 1e-8)
        

        key, subkey = random.split(key)

        perm = jax.random.permutation(subkey, T * B)

        flat_logp = flat_logp[perm]
        flat_states = flat_states[perm]
        flat_rewards = flat_rewards[perm]
        flat_V_ts = flat_V_ts[perm]
        flat_A_ts = flat_A_ts[perm]
        flat_ctrls = flat_ctrls[perm]

        Batch = {
            "states": flat_states,
            "logp_old": flat_logp,
            "rewards": flat_rewards,
            "V_targets": flat_V_ts,
            "advantages": flat_A_ts,
            "old_ctrls": flat_ctrls
        }

        return Batch, key
    
    @staticmethod
    @jax.jit
    def loss_fn(Model, V_ts, A_ts, old_log_p, states, old_ctrls, precent_done): #Work in batches

        x_in = jnp.concatenate([states, old_ctrls], -1)
        out = Model(x_in) # (B, 7)
        mus, log_stds, V_theta_ts = out[:, :3], out[:, 3:6], out[:, 6]

        stds = jnp.exp(log_stds)

        new_log_p = jnp.sum(-0.5 * (jnp.log(2 * jnp.pi * stds**2 + 1e-8) + ((old_ctrls - mus)**2) / (stds**2 + 1e-8)), -1) # (B)

        epsilon = .4
        L_P = -jnp.mean(jnp.minimum(jnp.exp(new_log_p - old_log_p) * A_ts, jnp.clip(jnp.exp(new_log_p - old_log_p), 1 - epsilon, 1 + epsilon) * A_ts))

        L_V = jnp.mean((V_theta_ts - V_ts)**2)

        L_E = jnp.mean(jnp.sum(.5*(jnp.log(2 * jnp.pi * stds**2 + 1e-8) + 1), axis=-1))

        c1 = .5
        c2 = 0
        # print(f"  L_P {L_P.item():7.3f} | L_V {L_V.item():7.3f} | H {L_E.item():5.2f}")

        return (precent_done)*L_P + c1 * L_V - c2 * L_E
    
    
    def train(self, epochs=100, minibatch_size=1024, ppo_epochs=1):

        loss_grad_fn = jax.jit(jax.value_and_grad(self.loss_fn))

        for epoch in range(epochs):

            # precent_done = jnp.minimum(epoch/(epochs*.1), 1)
            precent_done = 1
            batch, self.key = self.rollout(self.MLP, self.key, precent_done)

            # Print metrics from the collected data
            mean_reward = batch['rewards'].mean().item()
            print(batch["logp_old"].shape[0])
            print(f"Epoch {epoch:3d} | mean reward/step : {mean_reward:8.4f}")

            def _run_epoch(carry, xs):
                mlp_params, opt_state, batch, key = carry
                N = batch["logp_old"].shape[0]

                key, subkey = random.split(key)
                perm = jax.random.permutation(subkey, N)
                batch = jax.tree_util.tree_map(lambda x: x[perm], batch)

                def _run_mini_batch_epoch(carry, xs):
                    mlp_params, opt_state, batch = carry
                    j = xs
                    mini_batch = jax.tree_util.tree_map(lambda x: lax.dynamic_slice_in_dim(x, start_index=j, slice_size=minibatch_size, axis=0), batch)

                    loss, grads = loss_grad_fn(mlp_params, mini_batch["V_targets"], mini_batch["advantages"], mini_batch["logp_old"], mini_batch["states"], mini_batch["old_ctrls"], precent_done)
                    
                    updates, new_opt_state = self.opt.update(grads, opt_state, mlp_params)
                    new_mlp_params = optax.apply_updates(mlp_params, updates)
                    
                    return (new_mlp_params, new_opt_state, batch), loss
                
                mini_epoch_starting = jnp.arange(0, N - minibatch_size, minibatch_size)

                (mlp_params, opt_state, _), losses = jax.lax.scan(_run_mini_batch_epoch, (mlp_params, opt_state, batch), mini_epoch_starting, length=mini_epoch_starting.shape[0])

                return (mlp_params, opt_state, batch, key), jnp.mean(losses)
            
            (self.MLP, self.opt_state, _, self.key), losses = jax.lax.scan(_run_epoch, (self.MLP, self.opt_state, batch, self.key), None, length=ppo_epochs)
            
            # Print the mean loss from the *last* PPO epoch as a representative value
            print(f"          | mean loss: {losses[-1]:8.4f}")

            # ---- Checkpoint Saving ----
            ckpt_dir = Path("checkpoints").resolve()
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            checkpoints.save_checkpoint(
                ckpt_dir=ckpt_dir,
                target=self.MLP,
                step=epoch,
                prefix="policy_",
                overwrite=True
            )
            print(f"          | ✓ saved weights to {ckpt_dir}/policy_{epoch}")