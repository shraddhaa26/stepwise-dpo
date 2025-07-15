from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List

class StepwiseRewardModel:
    def __init__(self, model_name: str = "openai-community/gpt2"):
        """
        Initializes the stepwise reward model using a classification head.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def score_steps(self, steps: List[str]) -> List[float]:
        """
        Score each step in a reasoning chain.
        Returns a float score for each step (higher is better).
        """
        scores = []
        for step in steps:
            inputs = self.tokenizer(step, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            score = outputs.logits[0].mean().item()
            scores.append(score)
        return scores


# Optional test
if __name__ == "__main__":
    reward_model = StepwiseRewardModel()
    steps = [
        "The apple is red.",
        "It fell from the tree.",
        "The tree is tall and old.",
    ]
    scores = reward_model.score_steps(steps)
    print("Step scores:", scores)
