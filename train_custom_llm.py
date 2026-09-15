import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rag_engine import RAGEngine, BASE_DIR
from custom_llm import CustomSRMLLM, Tokenizer, MODEL_PATH, VOCAB_PATH


class LLMDataset(Dataset):
    def __init__(self, sequences, seq_len=64):
        self.seqs = []
        for seq in sequences:
            if len(seq) > 5:
                for i in range(0, len(seq) - seq_len, seq_len // 2):
                    chunk = seq[i:i + seq_len + 1]
                    if len(chunk) == seq_len + 1:
                        self.seqs.append(chunk)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        chunk = self.seqs[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def train_custom_llm(epochs=10, batch_size=32, lr=1e-3):
    print('=== Training Custom Neural Network Language Model (LLM) from Scratch ===')
    
    # 1. Load document corpus from rag_engine
    engine = RAGEngine()
    engine.build_or_load_index()
    texts = [c['text'] for c in engine.chunks]
    print(f'Loaded {len(texts)} document text chunks from corpus.')

    # 2. Build custom tokenizer vocabulary
    tokenizer = Tokenizer()
    tokenizer.build_vocab(texts, max_vocab_size=3000)
    tokenizer.save(VOCAB_PATH)
    print(f'Custom Vocabulary built ({len(tokenizer.vocab)} tokens).')

    # 3. Tokenize sequences
    encoded_seqs = [tokenizer.encode(t) for t in texts]
    dataset = LLMDataset(encoded_seqs, seq_len=64)
    print(f'Created {len(dataset)} training sequence blocks for causal language modeling.')

    if len(dataset) == 0:
        print('Dataset empty, creating expanded sequence blocks...')
        encoded_seqs = [tokenizer.encode(t) * 3 for t in texts]
        dataset = LLMDataset(encoded_seqs, seq_len=32)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. Instantiate Custom Neural LLM Architecture
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = len(tokenizer.vocab)
    model = CustomSRMLLM(vocab_size=vocab_size, embed_dim=128, num_heads=4, num_layers=3, ffn_dim=256, max_seq_len=128).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    print(f'Starting Neural Network Training on {device.upper()} ({epochs} epochs)...')
    t0 = time.time()
    for epoch in range(1, epochs + 1):
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
        if epoch % 2 == 0 or epoch == epochs:
            print(f'Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f}')

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
    train_custom_llm(epochs=10)
