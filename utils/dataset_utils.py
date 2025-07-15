from datasets import load_dataset
from typing import Dict, List

def load_prm_dataset(split="train"):
    """
    Loads a placeholder for PRM-like dataset.
    Replace 'intel/orca_dpo_pairs' with 'prm800k' if it becomes available.
    """
    dataset = load_dataset("Intel/orca_dpo_pairs", split=split)
    return dataset

def split_into_steps(answer: str) -> List[str]:
    """
    Splits a long reasoning answer into separate steps.
    Basic version: split by '.', 'then', 'so', etc.
    """
    import re
    steps = re.split(r'\.|\n|then|so', answer)
    steps = [s.strip() for s in steps if s.strip()]
    return steps

def prepare_stepwise_data(example: Dict[str, str]) -> Dict:
    """
    Prepares the data in stepwise format:
    - Breaks both chosen and rejected into steps
    - Returns a dict ready for reward scoring
    """
    chosen_steps = split_into_steps(example["chosen"])
    rejected_steps = split_into_steps(example["rejected"])

    return {
        "prompt": example["prompt"],
        "chosen_steps": chosen_steps,
        "rejected_steps": rejected_steps,
    }
