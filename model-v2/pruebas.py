import torch
from model import SovereignMoE, QriolloTokenizer
import json

# Load what we have
with open('data/dataset.jsonl', 'r', encoding='utf-8') as f:
    raw_data = "".join([json.loads(line)['text'] for line in f])
tokenizer = QriolloTokenizer(raw_data)

model = SovereignMoE(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load('saved/model_v2.pt'))
model.eval()

# Prompt the model
stop_id = tokenizer.stoi["<|end|>"]
prompt = "enchufar Chamullo el neg de Posta "
context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
generated = model.generate(context, max_new_tokens=300, stop_token_id=stop_id)[0].tolist()
context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
print(tokenizer.decode(generated))
