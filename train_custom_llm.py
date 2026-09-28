import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rag_engine import RAGEngine, BASE_DIR
from custom_llm import CustomSRMLLM, Tokenizer, MODEL_PATH, VOCAB_PATH


class LLMDataset(Dataset):
    def __init__(self, sequences, seq_len=64, stride=16):
        self.seqs = []
        all_tokens = []
        eos_id = 3
        for seq in sequences:
            if seq:
                all_tokens.extend(seq)
                if all_tokens[-1] != eos_id:
                    all_tokens.append(eos_id)

        for i in range(0, len(all_tokens) - seq_len, stride):
            chunk = all_tokens[i : i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.seqs.append(chunk)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        chunk = self.seqs[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def train_custom_llm(epochs=15, batch_size=64, lr=1e-3):
    print('=== Training Custom Neural Network Language Model (LLM) from Scratch ===')
    
    # 1. Load document corpus from rag_engine
    engine = RAGEngine()
    engine.build_or_load_index()
    texts = [c['text'] for c in engine.chunks]
    print(f'Loaded {len(texts)} document text chunks from corpus.')

    # 2. Build custom tokenizer vocabulary
    tokenizer = Tokenizer()
    tokenizer.build_vocab(texts, max_vocab_size=4000)
    tokenizer.save(VOCAB_PATH)
    print(f'Custom Vocabulary built ({len(tokenizer.vocab)} tokens).')

    # 3. Tokenize sequences into continuous stream with sliding window
    encoded_seqs = [tokenizer.encode(t) for t in texts]
    dataset = LLMDataset(encoded_seqs, seq_len=64, stride=48)
    print(f'Created {len(dataset)} training sequence blocks for causal language modeling.')

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 4. Instantiate Custom Neural LLM Architecture
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = len(tokenizer.vocab)
    model = CustomSRMLLM(vocab_size=vocab_size, embed_dim=128, num_heads=4, num_layers=3, ffn_dim=256, max_seq_len=128).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    print(f'Starting Neural Network Training on {device.upper()} (8 epochs, batch size 128)...')
    t0 = time.time()
    for epoch in range(1, 9):
        model.train()
        total_loss = 0.0
        batches = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        avg_loss = total_loss / max(batches, 1)
        print(f'Epoch {epoch:2d}/8 | Loss: {avg_loss:.4f}')

    t1 = time.time()
    print(f'Training completed in {t1-t0:.2f} seconds.')

    # 5. Save Model Weights
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Custom Model Weights saved to {MODEL_PATH.name}!')

    # 6. Test Autoregressive Generation
    prompt = 'SRMIST B.Tech CSE tuition fee'
    output = model.generate(tokenizer, prompt, max_new_tokens=40)
    print('\n--- Test Output from Your Custom Neural LLM ---')
    print(f'Prompt: {prompt}')
    print(f'Generated: {output}')


if __name__ == '__main__':
    train_custom_llm(epochs=15)
