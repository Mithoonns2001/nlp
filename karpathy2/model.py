import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers

# Preprocessing data
def preprocess_data(data: List[Tuple[str, str, str]]) -> Tuple[List[str], List[str]]:
    input_texts = []
    target_texts = []

    for desc, file_structure, code in data:
        input_texts.append(desc)
        target_texts.append(json.dumps({"file_structure": json.loads(file_structure), "code": json.loads(code)}))

    return input_texts, target_texts

# Training tokenizer
def train_tokenizer(input_texts: List[str], target_texts: List[str]) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(vocab_size=30000, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tokenizer.train_from_iterator(input_texts + target_texts, trainer=trainer)

    return tokenizer

# Dataset class
class ProjectGeneratorDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, input_texts: List[str], target_texts: List[str], block_size: int):
        self.tokenizer = tokenizer
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.block_size = block_size

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]

        input_tokens = self.tokenizer.encode(input_text).ids
        target_tokens = self.tokenizer.encode(target_text).ids

        # Truncate or pad input and target tokens
        input_tokens = input_tokens[:self.block_size]
        input_tokens += [0] * (self.block_size - len(input_tokens))

        target_tokens = target_tokens[:self.block_size]
        target_tokens += [0] * (self.block_size - len(target_tokens))

        return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)

# Your dataset
data = [
    ("Create a simple Python script that prints 'Hello, World!' with a separate file for the function",
     '{"root": ["main.py", "utils.py"]}',
     '{"main.py": "from utils import print_hello\\n\\nprint_hello()", "utils.py": "def print_hello():\\n    print(\'Hello, World!\')"}'),
    # ...
]

input_texts, target_texts = preprocess_data(data)
# Write preprocessed data to file
with open("preprocessed.txt", "w") as f:
    for input_text, target_text in zip(input_texts, target_texts):
        f.write(f"Input: {input_text}\n")
        f.write(f"Target: {target_text}\n")
        f.write("\n")
print("Preprocessed data saved to preprocessed.txt")

tokenizer = train_tokenizer(input_texts, target_texts)

block_size = 128
n_embd = 768
n_head = 12
n_layer = 6
dropout = 0.1

# Transformer model
class ProjectGeneratorModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head =        nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_tokens):
        token_embeddings = self.token_embedding_table(input_tokens)
        position_ids = torch.arange(input_tokens.shape[1], dtype=torch.long).unsqueeze(0).to(input_tokens.device)
        position_embeddings = self.position_embedding_table(position_ids)

        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd),
        )

    def forward(self, x):
        a = self.ln_1(x)
        x = x + self.attn(a, a, a)[0]
        x = x + self.mlp(self.ln_2(x))
        return x

# Model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProjectGeneratorModel(tokenizer.get_vocab_size(), n_embd, n_head, n_layer, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))

# Training loop
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for input_tokens, target_tokens in dataloader:
        input_tokens, target_tokens = input_tokens.to(device), target_tokens.to(device)

        optimizer.zero_grad()
        logits = model(input_tokens)
        loss = criterion(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

# Dataset and DataLoader
dataset = ProjectGeneratorDataset(tokenizer, input_texts, target_texts, block_size)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Train the model
n_epochs = 50
for epoch in range(n_epochs):
    loss = train_epoch(model, dataloader, optimizer, criterion, device)
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.4f}")



# Save the model
model_save_path = "project_generator_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Save the tokenizer
tokenizer_save_path = "project_generato_tokenizer"
tokenizer.save(tokenizer_save_path)
