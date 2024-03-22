import argparse

import DQN.DQNTrainer as DQNTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='PongNoFrameskip-v4', help="Name of the environment")
    parser.add_argument("--buffer_size", type=int, default=32, help="Size of the replay buffer")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.90, help="Exploration rate")
    parser.add_argument("--max_episode", type=int, default=10000, help="Maximum number of episodes")
    args = parser.parse_args()

    trainer = DQNTrainer.DQNTrainer(env_name=args.env_name,
                                    buffer_size=args.buffer_size,
                                    in_channels=args.in_channels,
                                    learning_rate=args.learning_rate,
                                    gamma=args.gamma,
                                    epsilon=args.epsilon)
    trainer.train(max_episode=args.max_episode)
    trainer.paint()