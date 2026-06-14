from __future__ import annotations

import argparse

from saegdnet.config import load_config
from saegdnet.training.engine import train_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_from_config(cfg)


if __name__ == "__main__":
    main()
