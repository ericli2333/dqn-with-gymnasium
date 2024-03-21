import DQN.DQNTrainer as DQNTrainer

if __name__ == "__main__":
    trainer = DQNTrainer.DQNTrainer(env_name='PongDeterministic-v4')
    trainer.train(max_episode=10000)
    trainer.paint()