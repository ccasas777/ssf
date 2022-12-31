import argparse
import json
import numpy as np
from metric.core import metric


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


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    print('Evaluate sensor performance')
    print(f"Use the following config to evaluate: {args.config}.")
    cfg = load_json(args.config)
    gt_idxs = np.loadtxt(cfg["gt_path"])
    det_idxs = np.loadtxt(cfg["det_path"])
    recall, precision = metric(gt_idxs, det_idxs)
    print('-' * 100)
    print("Evaluation results -> recall: {}; precision: {}".format(recall, precision))
