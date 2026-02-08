import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import re

def load_data(path):
    all_text = ""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Add a separator token so the model knows where one program ends
            all_text += data['text'] + "<|endoftext|>\n" 
    return all_text

import csv
import os

def log_loss(epoch, loss, expert_usage):
    file_exists = os.path.isfile('training_logs.csv')
    with open('./saved/training_logs.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        # Header for the first time
        if not file_exists:
            writer.writerow(['epoch', 'loss', 'exp1_pct', 'exp2_pct', 'exp3_pct', 'exp4_pct'])
        # Write the data
        writer.writerow([epoch, loss] + expert_usage)

class QriolloTokenizer:
    def __init__(self, text):
        self.special_token = "<|end|>"
        # Get unique chars, but make sure we don't include 
        # the characters that make up the special token by accident
        chars = sorted(list(set(text)))
        self.vocab = [self.special_token] + chars 
        self.vocab_size = len(self.vocab)
        
        self.stoi = { ch:i for i,ch in enumerate(self.vocab) }
        self.itos = { i:ch for i,ch in enumerate(self.vocab) }
        
        # This regex says: "Find the special token OR any single character"
        self.pattern = re.compile(r'(<\|end\|>|.)', re.DOTALL)

    def encode(self, s):
        tokens = self.pattern.findall(s)
        return [self.stoi[t] for t in tokens]
    
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

class Expert(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # The internal logic of the expert
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)

class MoELayer(nn.Module):
    def __init__(self, n_embd, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(n_embd, num_experts)
        self.experts = nn.ModuleList([Expert(n_embd) for _ in range(num_experts)])

    def forward(self, x):
        # x shape: (batch, seq_len, n_embd)
        logits = self.router(x)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            # Mask to find which tokens were assigned to this specific expert
            mask = (indices == i).any(dim=-1)
            if mask.any():
                # We weight the expert's output by the router's confidence
                expert_weight = weights[indices == i].unsqueeze(-1)
                out[mask] += expert(x[mask]) * expert_weight
        return out
    
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        # Key, Query, Value projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size() # Batch, Time (Sequence), Channels (Embedding)
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head processing
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
        # Apply causality (model can't see the future)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)    
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head, num_experts):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        # Instead of a single MLP, we use our MoE!
        self.moe = MoELayer(n_embd, num_experts)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.moe(self.ln_2(x))
        return x 
    
class SovereignMoE(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=6, num_experts=4):
        super().__init__()
        # Every token gets an embedding vector
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # Every position in the sequence (up to 1024) gets a vector
        self.position_embedding = nn.Embedding(1024, n_embd)
        
        # The "Spinal Column": A stack of Transformer Blocks
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, num_experts) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # Maps back to vocabulary

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Combine token and position info
        tok_emb = self.token_embedding(idx) 
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # Pass through the Experts
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Get the scores for the next token
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Shift logits and targets for cross-entropy calculation
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss    
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, stop_token_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -128:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Check if we hit the stop token
            if stop_token_id is not None and idx_next.item() == stop_token_id:
                break
                
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
        
    
if __name__ == "__main__":
    # 1. Setup a tiny version to test (128 embedding size, 4 experts)
    model = SovereignMoE(vocab_size=1000, n_embd=128, n_head=4, n_layer=2, num_experts=4)
    
    # 2. Fake some Qriollo input (Batch size 1, 8 tokens)
    dummy_input = torch.randint(0, 1000, (1, 8))
    
    # 3. Try a forward pass
    logits, _ = model(dummy_input)
    
    print(f"✅ Success! Brain is firing.")
    print(f"Output shape: {logits.shape} (Batch, Sequence, Vocab)") 
   
   
    