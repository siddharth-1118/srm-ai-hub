import os
import json
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'custom_srm_llm.pt'
VOCAB_PATH = BASE_DIR / 'custom_srm_vocab.json'


class Tokenizer:
    def __init__(self, vocab=None):
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def build_vocab(self, texts, max_vocab_size=3000):
        words = set()
        for t in texts:
            for w in t.split():
                words.add(w)
        for w in sorted(words):
            if w not in self.vocab and len(self.vocab) < max_vocab_size:
                idx = len(self.vocab)
                self.vocab[w] = idx
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        tokens = [self.vocab.get('<bos>', 2)]
        for w in text.split():
            tokens.append(self.vocab.get(w, self.vocab.get('<unk>', 1)))
        tokens.append(self.vocab.get('<eos>', 3))
        return tokens

    def decode(self, token_ids):
        words = []
        for tid in token_ids:
            w = self.inv_vocab.get(tid, '')
            if w not in ('<pad>', '<bos>', '<eos>', '<unk>'):
                words.append(w)
        return ' '.join(words)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return cls(vocab)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class CustomSRMLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=4, ffn_dim=512, max_seq_len=256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        mask = torch.full((T, T), float('-inf'), device=idx.device)
        mask = torch.triu(mask, diagonal=1)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, tokenizer, prompt, max_new_tokens=60, temperature=0.7):
        self.eval()
        tokens = tokenizer.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=next(self.parameters()).device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == tokenizer.vocab.get('<eos>', 3):
                break
            idx = torch.cat((idx, next_token), dim=1)

        return tokenizer.decode(idx[0].tolist())
