import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


tokenizer = tiktoken.get_encoding("gpt2")

with open("final_text.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

VOCAB_SIZE = tokenizer.n_vocab
EMBED_DIM  = 256


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids  = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Text too short for the requested max_length"

        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1:i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    tok = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tok, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers)


if __name__ == "__main__":
    print("\n\n________ DATA LOADING AND TOKENIZATION ________\n")
    enc_text = tokenizer.encode(raw_text)
    print("Tokenized length:", len(enc_text))

    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=10, stride=10, shuffle=False)
    data_iter  = iter(dataloader)

    inputs, targets = next(data_iter)
    print("\nFirst batch input  :", tokenizer.decode(inputs[0].tolist()))
    print("First batch target :", tokenizer.decode(targets[0].tolist()))

    inputs, targets = next(data_iter)
    print("\nSecond batch input :", tokenizer.decode(inputs[0].tolist()))
    print("Second batch target:", tokenizer.decode(targets[0].tolist()))

    print("\n\n________ EMBEDDINGS ________\n")
    print("Vocab size:", VOCAB_SIZE)

    MAX_LEN = 4
    dataloader  = create_dataloader_v1(raw_text, batch_size=8, max_length=MAX_LEN, stride=MAX_LEN, shuffle=False)
    inputs, _   = next(iter(dataloader))

    token_emb_layer = torch.nn.Embedding(VOCAB_SIZE, EMBED_DIM)
    pos_emb_layer   = torch.nn.Embedding(MAX_LEN, EMBED_DIM)

    token_embeddings = token_emb_layer(inputs)
    pos_embeddings   = pos_emb_layer(torch.arange(MAX_LEN))
    input_embeddings = token_embeddings + pos_embeddings

    print("Token embedding shape    :", token_embeddings.shape)
    print("Positional embedding shape:", pos_embeddings.shape)
    print("Input embedding shape    :", input_embeddings.shape)
    print("\nSample token embedding (batch 0):\n", input_embeddings[0])
