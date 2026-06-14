"""
main.py
=======
One command to run the whole pipeline end-to-end:

    python main.py            # download data -> train -> print a final sample

It is just a thin wrapper around the pieces in data.py / model.py / train.py /
generate.py so you can also use those individually (see each file's __main__).
"""

import torch

from config import get_config
from train import train
from generate import load_for_generation, generate


def main() -> None:
    cfg = get_config()

    # 1. Train (this downloads the data on first run and saves the best model).
    train(cfg)

    # 2. Reload the best checkpoint and print a longer final sample.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_for_generation(cfg, device)
    text = generate(model, tokenizer, device, prompt="It was ",
                    length=cfg.gen_length, temperature=cfg.temperature,
                    top_k=cfg.top_k)
    print("\n" + "=" * 70)
    print("FINAL SAMPLE")
    print("=" * 70)
    print(text)
    print("=" * 70)


if __name__ == "__main__":
    main()
