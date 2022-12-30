import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple


class Metric:

    def __call__(self, gt_idxs: List[int], det_idxs: List[int]) -> Tuple[float, float]:
        """
            Feed into the known ground truth and detected peak idxs from one's scan
            Args:
                gt_idxs: list of idxs
                det_idxs: list of idxs


        """
        TP, FP, FN = [], [], []
        gt_idxs = np.asarray(gt_idxs)
        det_idxs = np.asarray(det_idxs)

        gt_idxs = gt_idxs[:, None]
        det_idxs = det_idxs[:, None]
        cost_matrics = abs(gt_idxs - det_idxs.T)
        rows, cols = linear_sum_assignment(-cost_matrics, maximize=True)
        costs = cost_matrics[rows, cols]
        pairs, picked = [], []
        zone = 500
        for r, c, cost in zip(rows, cols, costs):
            if cost < zone:
                if c not in picked:
                    pairs.append([r, c])
                    picked.append(c)

        TP = len(pairs)
        FN = np.shape(gt_idxs)[0] - TP
        FP = det_idxs.shape[0] - TP
        return self.report(TP, FN, FP)

    def report(self, TP, FN, FP):
        epsilon = 1e-9
        recall = TP / max(TP + FN, epsilon)
        precision = TP / max(TP + FP, epsilon)
        return recall, precision
