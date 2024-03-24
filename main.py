import argparse

import DQN.DQNTrainer as DQNTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='PongNoFrameskip-v4', help="Name of the environment")
    parser.add_argument("--buffer_size", type=int, default=32, help="Size of the replay buffer")
    parser.add_argument("--in_channels", type=int, default=4, help="Number of input channels")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.90, help="Exploration rate")
    parser.add_argument("--max_episode", type=int, default=int(1e9), help="Maximum number of episodes")
    parser.add_argument("--log_level", type=int, default=1, help="1: tensor board only\n2: debug only")
    args = parser.parse_args()

    trainer = DQNTrainer.DQNTrainer(env_name=args.env_name,
                                    buffer_size=args.buffer_size,
                                    in_channels=args.in_channels,
                                    learning_rate=args.learning_rate,
                                    gamma=args.gamma,
                                    epsilon=args.epsilon,
                                    log_level=args.log_level
                                    )
    trainer.train(max_frame=args.max_episode)
    trainer.paint()