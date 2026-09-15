import euclid
import numpy as np

from archid.reinforcement_learning.algorithms.actor_critic.v_critic.v_critic import VCritic


class PPO(VCritic):

    def __init__(
        self,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        epochs=10,
        minibatch_size=256,
        value_clip_epsilon=0.2,
        max_grad_norm=0.5,
        target_kl=0.03,
    ):
        super().__init__(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            gamma=gamma,
        )

        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.value_clip_epsilon = value_clip_epsilon
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

    # ============================================================
    # POLICY
    # ============================================================

    def predict(self, state):
        distribution = self.actor.forward(state)
        return distribution.sample()

    # ============================================================
    # HELPERS
    # ============================================================

    def _get_parameters(self, network):
        try:
            return network.parameters()
        except Exception:
            return None

    def _clip_gradients(self, optimizer, parameters):
        if self.max_grad_norm is None or self.max_grad_norm <= 0:
            return 0.0

        grads = None

        if hasattr(optimizer, "gradients"):
            try:
                grads = optimizer.gradients
            except Exception:
                pass

        if grads is None and hasattr(optimizer, "grads"):
            try:
                grads = optimizer.grads
            except Exception:
                pass

        if grads is None and parameters is not None:
            grads = []

            for parameter in parameters:
                grad = getattr(parameter, "grad", None)

                if grad is not None:
                    grads.append(grad)

            if not grads:
                grads = None

        if grads is None:
            return 0.0

        if isinstance(grads, dict):
            grad_list = list(grads.values())
        elif isinstance(grads, (list, tuple)):
            grad_list = list(grads)
        else:
            grad_list = [grads]

        arrays = []

        for grad in grad_list:
            data = grad.data if hasattr(grad, "data") else grad

            try:
                arrays.append(
                    np.asarray(data, dtype=np.float32)
                )
            except Exception:
                pass

        if not arrays:
            return 0.0

        total_norm = float(
            np.sqrt(
                sum(
                    float((array ** 2).sum())
                    for array in arrays
                )
            )
        )

        if total_norm <= self.max_grad_norm:
            return total_norm

        scale = self.max_grad_norm / (total_norm + 1e-8)

        for grad in grad_list:
            try:
                if hasattr(grad, "data"):
                    grad.data = (
                        np.asarray(grad.data)
                        * scale
                    ).astype(np.float32)

                elif isinstance(grad, np.ndarray):
                    grad *= scale

            except Exception:
                pass

        return total_norm

    # ============================================================
    # GAE
    # ============================================================

    def compute_advantages(
        self,
        rewards,
        values,
        next_values,
        dones,
    ):
        rewards_np = np.asarray(
            rewards.data,
            dtype=np.float32
        ).reshape(-1)

        values_np = np.asarray(
            values.data,
            dtype=np.float32
        ).reshape(-1)

        next_values_np = np.asarray(
            next_values.data,
            dtype=np.float32
        ).reshape(-1)

        advantages = np.zeros_like(rewards_np)

        gae = 0.0

        for t in reversed(range(len(rewards_np))):

            if dones[t]:
                next_non_terminal = 0.0
            else:
                next_non_terminal = 1.0

            delta = (
                rewards_np[t]
                + self.gamma
                * next_values_np[t]
                * next_non_terminal
                - values_np[t]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * next_non_terminal
                * gae
            )

            advantages[t] = gae

        return euclid.Tensor(advantages)

    # ============================================================
    # UPDATE
    # ============================================================

    def update(self, experience):

        if len(experience) == 0:
            return {}

        # --------------------------------------------------------
        # Build rollout tensors
        # --------------------------------------------------------

        states_np = np.stack([
            np.asarray(
                transition.state.data,
                dtype=np.float32
            )
            for transition in experience
        ])

        actions_np = np.asarray([
            transition.action
            for transition in experience
        ], dtype=np.float32)

        old_log_probs_np = np.asarray([
            transition.log_prob
            for transition in experience
        ], dtype=np.float32).reshape(-1)

        rewards_np = np.asarray([
            transition.reward
            for transition in experience
        ], dtype=np.float32).reshape(-1)

        next_states_np = np.stack([
            np.asarray(
                transition.next_state.data,
                dtype=np.float32
            )
            for transition in experience
        ])

        dones = [
            bool(transition.done)
            for transition in experience
        ]

        states = euclid.Tensor(states_np)
        actions = euclid.Tensor(actions_np)
        old_log_probs = euclid.Tensor(old_log_probs_np)
        rewards = euclid.Tensor(rewards_np)
        next_states = euclid.Tensor(next_states_np)

        # --------------------------------------------------------
        # OLD critic predictions
        #
        # These must stay fixed during PPO's multiple epochs.
        # --------------------------------------------------------

        old_values = self.critic.forward(states)

        next_values = self.critic.forward(next_states)

        old_values_np = np.asarray(
            old_values.data,
            dtype=np.float32
        ).reshape(-1)

        next_values_np = np.asarray(
            next_values.data,
            dtype=np.float32
        ).reshape(-1)

        # --------------------------------------------------------
        # GAE
        # --------------------------------------------------------

        advantages = self.compute_advantages(
            rewards,
            euclid.Tensor(old_values_np),
            euclid.Tensor(next_values_np),
            dones,
        )

        advantages_np = np.asarray(
            advantages.data,
            dtype=np.float32
        ).reshape(-1)

        # --------------------------------------------------------
        # Returns
        # --------------------------------------------------------

        returns_np = (
            old_values_np
            + advantages_np
        )

        # --------------------------------------------------------
        # Advantage normalization
        # --------------------------------------------------------

        advantages_np = (
            advantages_np
            - advantages_np.mean()
        ) / (
            advantages_np.std() + 1e-8
        )

        # --------------------------------------------------------
        # Freeze rollout data
        # --------------------------------------------------------

        old_log_probs_np = old_log_probs_np.copy()
        old_values_np = old_values_np.copy()
        returns_np = returns_np.copy()
        advantages_np = advantages_np.copy()

        n = len(experience)

        actor_parameters = self._get_parameters(
            self.actor
        )

        critic_parameters = self._get_parameters(
            self.critic
        )

        # --------------------------------------------------------
        # Statistics
        # --------------------------------------------------------

        actor_losses = []
        critic_losses = []
        entropy_values = []
        kl_values = []
        clip_fractions = []

        # ========================================================
        # PPO EPOCHS
        # ========================================================

        for epoch in range(self.epochs):

            indices = np.random.permutation(n)

            epoch_kl = 0.0
            epoch_count = 0

            # ====================================================
            # MINIBATCHES
            # ====================================================

            for start in range(
                0,
                n,
                self.minibatch_size
            ):

                batch_indices = indices[
                    start:
                    start + self.minibatch_size
                ]

                batch_states = euclid.Tensor(
                    states_np[batch_indices]
                )

                batch_actions = euclid.Tensor(
                    actions_np[batch_indices]
                )

                batch_old_log_probs = euclid.Tensor(
                    old_log_probs_np[batch_indices]
                )

                batch_advantages = euclid.Tensor(
                    advantages_np[batch_indices]
                )

                batch_returns = euclid.Tensor(
                    returns_np[batch_indices]
                )

                batch_old_values = euclid.Tensor(
                    old_values_np[batch_indices]
                )

                # =================================================
                # ACTOR
                # =================================================

                distribution = self.actor.forward(
                    batch_states
                )

                new_log_probs = distribution.log_prob(
                    batch_actions
                )

                entropy = distribution.entropy().mean()

                # PPO ratio:
                #
                # exp(new_log_prob - old_log_prob)
                #

                log_ratio = (
                    new_log_probs
                    - batch_old_log_probs
                )

                ratio = log_ratio.exp()

                clipped_ratio = ratio.clip(
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                )

                surrogate_1 = (
                    ratio
                    * batch_advantages
                )

                surrogate_2 = (
                    clipped_ratio
                    * batch_advantages
                )

                surrogate = surrogate_1.minimum(
                    surrogate_2
                )

                policy_loss = (
                    -surrogate.mean()
                    - self.entropy_coef * entropy
                )

                self.actor_optimizer.zero_grad()

                policy_loss.backward()

                self._clip_gradients(
                    self.actor_optimizer,
                    actor_parameters,
                )

                self.actor_optimizer.step()

                # =================================================
                # CRITIC
                # =================================================

                new_values = self.critic.forward(
                    batch_states
                )

                new_values_flat = new_values.reshape(-1)

                # -------------------------------------------------
                # PPO clipped value prediction
                # -------------------------------------------------

                value_delta = (
                    new_values_flat
                    - batch_old_values
                )

                clipped_values = (
                    batch_old_values
                    + value_delta.clip(
                        -self.value_clip_epsilon,
                        self.value_clip_epsilon,
                    )
                )

                value_loss_unclipped = (
                    new_values_flat
                    - batch_returns
                ) ** 2

                value_loss_clipped = (
                    clipped_values
                    - batch_returns
                ) ** 2

                value_loss = value_loss_unclipped.maximum(
                    value_loss_clipped
                ).mean()

                critic_loss = (
                    self.value_coef
                    * value_loss
                )

                self.critic_optimizer.zero_grad()

                critic_loss.backward()

                self._clip_gradients(
                    self.critic_optimizer,
                    critic_parameters,
                )

                self.critic_optimizer.step()

                # =================================================
                # METRICS
                # =================================================

                ratio_np = np.asarray(
                    ratio.data,
                    dtype=np.float32
                ).reshape(-1)

                log_ratio_np = np.asarray(
                    log_ratio.data,
                    dtype=np.float32
                ).reshape(-1)

                clipped_np = (
                    np.abs(
                        ratio_np - 1.0
                    )
                    > self.clip_epsilon
                )

                approx_kl = float(
                    -log_ratio_np.mean()
                )

                clip_fraction = float(
                    clipped_np.mean()
                )

                actor_losses.append(
                    float(
                        np.asarray(
                            policy_loss.data
                        ).mean()
                    )
                )

                critic_losses.append(
                    float(
                        np.asarray(
                            critic_loss.data
                        ).mean()
                    )
                )

                entropy_values.append(
                    float(
                        np.asarray(
                            entropy.data
                        ).mean()
                    )
                )

                kl_values.append(approx_kl)
                clip_fractions.append(
                    clip_fraction
                )

                epoch_kl += approx_kl
                epoch_count += 1

            # ====================================================
            # EARLY STOPPING
            # ====================================================

            if (
                self.target_kl is not None
                and epoch_count > 0
            ):

                mean_epoch_kl = (
                    epoch_kl / epoch_count
                )

                if mean_epoch_kl > (
                    1.5 * self.target_kl
                ):
                    break

        # ========================================================
        # RETURN DIAGNOSTICS
        # ========================================================

        return {
            "policy_loss": float(
                np.mean(actor_losses)
            ) if actor_losses else 0.0,

            "value_loss": float(
                np.mean(critic_losses)
            ) if critic_losses else 0.0,

            "entropy": float(
                np.mean(entropy_values)
            ) if entropy_values else 0.0,

            "approx_kl": float(
                np.mean(kl_values)
            ) if kl_values else 0.0,

            "clip_fraction": float(
                np.mean(clip_fractions)
            ) if clip_fractions else 0.0,

            "advantage_mean": float(
                advantages_np.mean()
            ),

            "advantage_std": float(
                advantages_np.std()
            ),

            "value_mean": float(
                old_values_np.mean()
            ),
        }