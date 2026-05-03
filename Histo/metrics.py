"""Metrics for model evaluation."""
import numpy as np


class DiceAccumulator:
    """Accumulates intersection/union counts per class for Dice calculation.
    Uses O(num_classes) memory regardless of dataset size."""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.intersection = np.zeros(num_classes, dtype=np.float64)
        self.pred_sum = np.zeros(num_classes, dtype=np.float64)
        self.target_sum = np.zeros(num_classes, dtype=np.float64)
    
    def update(self, predictions, targets):
        """Update running stats with a batch. Tensors are moved to CPU internally."""
        preds = predictions.cpu().numpy()
        tgts = targets.cpu().numpy()
        for c in range(self.num_classes):
            p = (preds == c)
            t = (tgts == c)
            self.intersection[c] += np.sum(p & t)
            self.pred_sum[c] += np.sum(p)
            self.target_sum[c] += np.sum(t)
    
    def compute(self):
        """Compute final Dice scores.
        Returns: (dice_per_class list, mean_dice float)"""
        dice_per_class = []
        for c in range(self.num_classes):
            union = self.pred_sum[c] + self.target_sum[c]
            if union == 0:
                dice_per_class.append(1.0)
            else:
                dice_per_class.append(2.0 * self.intersection[c] / union)
        return dice_per_class, float(np.mean(dice_per_class))
    
    def reset(self):
        """Reset all counters."""
        self.intersection.fill(0)
        self.pred_sum.fill(0)
        self.target_sum.fill(0)
