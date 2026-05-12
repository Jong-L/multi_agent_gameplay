from ray.rllib.algorithms.ppo import PPOConfig

# Create a config instance for the PPO algorithm.
config: PPOConfig = (
    PPOConfig()
    .environment("Pendulum-v1")
)
config.env_runners(num_env_runners=2)
config.training(
    lr=0.0002,
    train_batch_size_per_learner=2000,
    num_epochs=10,
)

ppo=config.build_algo()

