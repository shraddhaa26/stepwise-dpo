from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_path = "./outputs"  # adjust if your model is saved elsewhere
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = "Why is the sky blue? Give a step-by-step explanation."
    response = generate_response(model, tokenizer, prompt)

    print("\n🧠 Model's Step-by-Step Response:\n")
    print(response)

if __name__ == "__main__":
    main()
