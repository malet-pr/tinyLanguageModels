import torch
import torch.optim as optim

# 1. Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YourMoEModel().to(device) # We'll put the full class together
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# 2. The Loop
for epoch in range(max_epochs):
    optimizer.zero_grad()
    
    # x: input tokens, y: target tokens (next word)
    logits, loss = model(x, y)
    
    loss.backward()
    
    # SAFETY RAIL: Prevent the 11,000 epoch explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # 3. Checkpoint (Save your work!)
    if epoch % 500 == 0:
        torch.save(model.state_state_dict(), f"checkpoint_{epoch}.pt")
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")
        
