import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from model import ProjectGeneratorModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_embd = 768
n_head = 12
n_layer = 12
dropout = 0.1

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the model
model_save_path = "C:/Users/NAGARAJAN K/Desktop/initial/project_generator_model.pth"
model = ProjectGeneratorModel(tokenizer.vocab_size, n_embd, n_head, n_layer, dropout).to(device)

model.load_state_dict(torch.load(model_save_path))
model.eval()
print("Model loaded.")

def generate_project(description):
    # Tokenize input description
    input_tokens = tokenizer.encode(description, return_tensors="pt").to(device)

    # Generate file structure and code
    output_tokens = model.generate(input_tokens, max_length=1024, num_return_sequences=1, no_repeat_ngram_size=3)
    generated_text = tokenizer.decode(output_tokens[0])

    return generated_text

# Provide a description
description = "Create a simple Python script that prints 'Hello, World!' with a separate file for the function"

# Generate the project file structure and code
result = generate_project(description)
print(result)
