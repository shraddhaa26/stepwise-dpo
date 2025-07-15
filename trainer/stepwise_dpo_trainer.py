from trl import DPOTrainer
import torch
from torch.nn import functional as F
from typing import Any, Dict

class StepwiseDPOTrainer(DPOTrainer):
    def compute_loss(self, model, inputs: Dict[str, Any], return_outputs=False):
        """
        Custom loss function using stepwise rewards.
        This overrides DPOTrainer’s loss to use step-level chosen vs rejected scores.
        """

        # 1. Get rewards from input
        stepwise_chosen_rewards = torch.tensor(inputs["stepwise_chosen_rewards"], dtype=torch.float32)
        stepwise_rejected_rewards = torch.tensor(inputs["stepwise_rejected_rewards"], dtype=torch.float32)

        # 2. Convert step lists into one reward (mean)
        r_chosen = stepwise_chosen_rewards.mean(dim=1)
        r_rejected = stepwise_rejected_rewards.mean(dim=1)

        # 3. Compute stepwise DPO loss
        beta = 0.1  # you can tune this
        logits = beta * (r_chosen - r_rejected)

        # 4. Final loss = negative log-sigmoid of logits
        loss = -F.logsigmoid(logits).mean()

        if return_outputs:
            return loss, {"logits": logits}
        return loss
