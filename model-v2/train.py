import torch
import os
import json
from model import SovereignMoE, QriolloTokenizer

# 1. Setup Folders
os.makedirs('saved', exist_ok=True)

# 2. Load Data & Tokenizer
with open('data/dataset.jsonl', 'r', encoding='utf-8') as f:
    # Use json.loads to ensure we get the actual 'á' and not the '\u00f3' string
    raw_data = "".join([json.loads(line)['text'] for line in f])
tokenizer = QriolloTokenizer(raw_data)     

# Save the vocab immediately - The "Sovereign" Dictionary
with open('saved/vocab.json', 'w', encoding='utf-8') as f:
    json.dump({"stoi": tokenizer.stoi, "itos": tokenizer.itos}, f)

# 3. Hyperparameters
batch_size = 16 
block_size = 128  
max_iters = 2000
learning_rate = 1e-3
eval_interval = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'


data = torch.tensor(tokenizer.encode(raw_data), dtype=torch.long)
model = SovereignMoE(vocab_size=tokenizer.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Feeding {len(data)} characters into MoE...")
print(f"Vocab size: {tokenizer.vocab_size} | Device: {device}")

# 4. Training Loop
for iter in range(max_iters):
    # Get batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)

    # Forward pass
    logits, loss = model(x, y)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        print(f"Step {iter}: Loss {loss.item():.4f}")
        # Save checkpoint
        torch.save(model.state_dict(), 'saved/model_v2.pt')

print("Training Complete. Final Loss:", loss.item())

