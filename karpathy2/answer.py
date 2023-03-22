import json
import torch
from tokenizers import Tokenizer
from typing import Dict, Tuple
from model import ProjectGeneratorModel
import torch.nn as nn
from typing import List, Tuple

block_size = 128
n_embd = 768
n_head = 12
n_layer = 6
dropout = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def generate_project_structure_and_code(model: nn.Module, tokenizer: Tokenizer, description: str, block_size: int) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    # Prepare the input
    input_tokens = tokenizer.encode(description).ids
    input_tokens = input_tokens[:block_size]
    input_tokens += [0] * (block_size - len(input_tokens))
    input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
    
    # Generate the output using the model
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
    
    # Decode the logits into tokens using greedy decoding
    output_tokens = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    
    # Decode the tokens into text
    output_text = tokenizer.decode(output_tokens)
    
    # Parse the generated text into file structure and code snippets
    output_dict = json.loads(output_text)
    file_structure = output_dict["file_structure"]
    code = {k: v for k, v in output_dict["code"].items()}
    
    return file_structure, code



# Load the tokenizer
tokenizer_path = "C:/Users/NAGARAJAN K/Desktop/initial/project_generato_tokenizer"
tokenizer = Tokenizer.from_file(tokenizer_path)

# Load the trained model
model_save_path = "C:/Users/NAGARAJAN K/Desktop/initial/project_generator_model.pth"
model = ProjectGeneratorModel(tokenizer.get_vocab_size(), n_embd, n_head, n_layer, dropout).to(device)
model.load_state_dict(torch.load(model_save_path))

# Generate project file structure and code for a new description
description = "Create a simple Python script that prints 'Hello, World!' with a separate file for the function"
block_size = 128
file_structure, code = generate_project_structure_and_code(model, tokenizer, description, block_size)

print("File structure:", file_structure)
print("Code:", code)
