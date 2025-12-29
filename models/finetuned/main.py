import yaml
from src.train.trainer import run_training

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    run_training(config)