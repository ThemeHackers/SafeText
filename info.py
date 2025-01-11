import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.nn.functional import softmax
import numpy as np

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    return tokenizer, model, config

def print_model_details(model, config):
    print("Model Structure:")
    print(model)
    print("\nModel Parameters:")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Requires Grad: {param.requires_grad}")

    print("\nModel Configuration:")
    print(config)
    print("\nConfiguration Details:")
    print(f"Number of labels: {config.num_labels}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Model type: {config.model_type}")

def tokenize_input(text, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    decoded_text = tokenizer.decode(inputs["input_ids"][0])
    return inputs, tokens, decoded_text

def predict(inputs, model):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        logits = outputs.logits
        probs = softmax(logits, dim=-1)
        predicted_label = torch.argmax(probs, dim=-1).item()
    return logits, probs, predicted_label, outputs

def model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-Trainable Parameters: {total_params - trainable_params:,}")

def calculate_model_size(model):
    param_size = sum(p.element_size() * p.numel() for p in model.parameters())
    size_in_mb = param_size / (1024 * 1024)  
    print(f"\nModel Size: {size_in_mb:.2f} MB")
    return size_in_mb

def calculate_parameter_statistics(model):
    for name, param in model.named_parameters():
        
        flattened_param = param.detach().cpu().numpy().flatten()
        
  
        mean = np.mean(flattened_param)
        std = np.std(flattened_param)
        
        print(f"\nStatistics for Parameter: {name}")
        print(f"Mean: {mean:.4f}, Standard Deviation: {std:.4f}")

def main():
    model_path = "./fine_tuning_model"
    text = "This is a test message for the model."  

    tokenizer, model, config = load_model_and_tokenizer(model_path)
    
    print_model_details(model, config)
    model_size(model)
    calculate_model_size(model)
    
    inputs, tokens, decoded_text = tokenize_input(text, tokenizer)
    print("\nTokenized Input:", inputs)
    print("\nTokens:", tokens)
    print("Decoded Text:", decoded_text)
    
    logits, probs, predicted_label, outputs = predict(inputs, model)
    print("\nLogits:", logits)
    print("\nProbabilities:", probs)
    print("\nPredicted Label:", predicted_label)
    
    if outputs.hidden_states:
        print("\nNumber of Hidden States:", len(outputs.hidden_states))
    else:
        print("\nHidden States not available.")
    
    if outputs.attentions:
        print("Number of Attention Layers:", len(outputs.attentions))
    else:
        print("Attention Layers not available.")
    
    state_dict = model.state_dict()
    print("\nModel Weights:")
    for name, param in state_dict.items():
        print(f"Parameter: {name} | Shape: {param.shape}")
    

    calculate_parameter_statistics(model)

if __name__ == "__main__":
    main()
