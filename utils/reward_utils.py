from typing import Dict, List
from models.stepwise_reward_model import StepwiseRewardModel
from utils.dataset_utils import prepare_stepwise_data

# Global reward model instance (for efficiency)
reward_model = StepwiseRewardModel()

def score_steps(steps: List[str]) -> List[float]:
    """
    Uses the global reward_model to score every reasoning step.
    """
    return reward_model.score_steps(steps)

def aggregate(scores: List[float], method: str = "mean") -> float:
    """
    Combines step scores into a single reward per answer.
    You can change to 'sum', 'max', etc.
    """
    if method == "mean":
        return float(sum(scores) / len(scores))
    elif method == "sum":
        return float(sum(scores))
    elif method == "max":
        return float(max(scores))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

def score_example(example: Dict[str, str]) -> Dict:
    """
    Takes a raw dataset example, splits into steps,
    scores each step, aggregates reward,
    and returns all data needed by the trainer.
    """
    formatted = prepare_stepwise_data(example)

    chosen_scores = score_steps(formatted["chosen_steps"])
    rejected_scores = score_steps(formatted["rejected_steps"])

    agg_chosen = aggregate(chosen_scores)
    agg_rejected = aggregate(rejected_scores)

    return {
        "prompt": example["prompt"],
        "chosen_steps": formatted["chosen_steps"],
        "rejected_steps": formatted["rejected_steps"],
        "stepwise_chosen_rewards": chosen_scores,
        "stepwise_rejected_rewards": rejected_scores,
        "agg_chosen_reward": agg_chosen,
        "agg_rejected_reward": agg_rejected,
    }
