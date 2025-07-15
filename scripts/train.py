from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trainer.stepwise_dpo_trainer import StepwiseDPOTrainer
from utils.reward_utils import score_example

def main():
    # 1. Load dataset (1% for quick test)
    raw_dataset = load_dataset("Intel/orca_dpo_pairs", split="train[:1%]")
    scored_dataset = raw_dataset.map(score_example)

    # 2. Load model & tokenizer
    model_name = "tiiuae/falcon-rw-1b"  # or use "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 3. Set training args
    training_args = TrainingArguments(
        output_dir="./outputs/",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        logging_steps=5,
        save_steps=50,
        evaluation_strategy="no",
        logging_dir="./logs",
        report_t_
