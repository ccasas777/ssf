import numpy as np
import os
import argparse
import json
from scipy.optimize import linear_sum_assignment


def load_json(path):
    """The function of loading json file

    Arguments:
        path {str} -- The path of the json file

    Returns:
        list, dict -- The obj stored in the json file
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


class Eval:

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self):
        gt_idxs = np.loadtxt(self.cfg["gt_path"])
        det_idxs = np.loadtxt(self.cfg["det_path"])
        TP, FP, FN = [], [], []
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


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    print('Evaluate sensor performance')
    print(f"Use the following config to evaluate: {args.config}.")
    cfg = load_json(args.config)
    eval = Eval(cfg)
    recall, precision = eval()
    print('-' * 100)
    print("Evaluation results -> recall: {}; precision: {}".format(recall, precision))
