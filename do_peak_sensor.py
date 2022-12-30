from locale import normalize
import os
from click import core
import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from utils.io import load_json
from tqdm import tqdm
from glob import glob
from metric.core import metric
import argparse


class Sensor:

    def __init__(self, config):
        self.config = config
        self.pzt = self.config['pzt']
        self.gt_root_dir = self.config['gt_root_dir']
        self.pzt_ind = np.loadtxt(self.pzt['index_path'])

    def __call__(self):
        n, peak_dist, threshold = 10, 600, 0.1
        print('-' * 100)
        print("get pzt {} cycles".format(len(self.pzt_ind)))
        print("data size is %d" % (self.pzt_ind[0][1] - self.pzt_ind[0][0]))
        raw_data = self.get_scan(self.pzt['scan_path'], n)
        assert n <= len(
            raw_data), "Please check raw data length. You assigned n: {} number, but raw-data scans only {}".format(n, len(
                raw_data))
        data_size = self.data_size_check(raw_data)
        if (data_size == -1):
            print("data size error")
            return -1
        kernel, peak_dist = self.warm_up(n, raw_data)
        ds_gen = self.get_peaks(kernel, raw_data, peak_dist, threshold)
        return np.asarray(ds_gen)

    def warm_up(self, n, raw_data):
        """
            Determine how many rounds
        """
        k_size = 500
        up_gt_idxs = np.loadtxt(os.path.join(
            self.gt_root_dir, 'up', 'gt_idxs.txt'))
        up_scans = self.get_scan(os.path.join(
            self.gt_root_dir, 'up'), n=1)
        self.down_gt_idxs = np.loadtxt(os.path.join(
            self.gt_root_dir, 'down', 'gt_idxs.txt'))
        down_scans = self.get_scan(os.path.join(
            self.gt_root_dir, 'down'), n=1)
        scs = []
        for i in range(1, n):
            kernel = self.create_kernel(i, raw_data, k_size, method='argmax')
            peaks = self.get_peaks(kernel, up_scans, peak_dist=600)
            corrected_idxs = [np.argmax(
                up_scans[0][p - 500:p + 500]) + p - 500 for p in peaks]
            _, _, F1 = metric(up_gt_idxs, corrected_idxs)
            scs.append(F1)
        iterative_n = np.argmax(scs) + 1
        kernel = self.create_kernel(
            iterative_n, raw_data, k_size, method='argmax')
        scs = []
        for i in range(1, 11):
            peaks = self.get_peaks(kernel, up_scans, peak_dist=i * 100)
            corrected_idxs = [np.argmax(
                up_scans[0][p - 500:p + 500]) + p - 500 for p in peaks]
            _, _, F1 = metric(up_gt_idxs, corrected_idxs)
            scs.append(F1)
        peak_dist = (np.argmax(scs) + 1) * 100
        return kernel, peak_dist

    def create_kernel(self, n, raw_data,  size, method='argmax'):
        kernel = np.zeros(2 * size)
        mute = 5000
        data_size = len(raw_data[0])
        for i in range(n):
            cent = np.argmax(raw_data[i])
            tmp = raw_data[i][cent - size:cent + size]
            tmp = self.normalize(tmp)
            kernel = kernel + tmp
            # tmp = np.correlate(raw_data[i], kernel, 'same')
            # tmp = self.normalize(tmp)
            # tmp[0:mute] = 0
            # tmp[data_size - mute:data_size] = 0
            # peaks, _ = find_peaks(tmp, height=0.1, distance=600)
            # peak_vals = tmp[peaks]
            # valid_peak_mask = peak_vals > 0.5
            # peaks = peaks[valid_peak_mask]
            # scan_kernel = []
            # for p in peaks:
            #     tmp = raw_data[i][p - size:p + size]
            #     tmp = self.normalize(tmp)
            #     scan_kernel.append(tmp)
        return self.normalize(kernel)

    def get_peaks(self, kernel, raw_data, peak_dist, threshold=0.1):
        mute = 5000
        data_size = np.size(raw_data[0])
        ds_gen = []
        for round_raw_data in raw_data:
            # round_raw_data = self.normalize(round_raw_data)
            corr_tmp = np.correlate(round_raw_data, kernel, 'same')
            corr_tmp = self.normalize(corr_tmp)
            corr_tmp[:mute] = 0
            corr_tmp[data_size - mute:data_size] = 0
            peaks, _ = find_peaks(
                corr_tmp, height=threshold, distance=peak_dist)
            ds_gen.append(peaks)

        return peaks

    def get_scan(self, path, n):
        raw_data = []
        for i in range(n):
            tmp = np.loadtxt(os.path.join(path, str(i) + ".txt"))
            raw_data.append(tmp)
        return raw_data

    def normalize(self, data):
        peak_ind = np.argmax(data)
        peak_val = data[peak_ind]
        min_val = np.min(data)
        data = (data - min_val) / (peak_val - min_val)
        return data

    def data_size_check(self, raw_data):
        prev_l = len(raw_data[0])
        check_ok = 0
        for i in range(len(raw_data) - 1):
            if (len(raw_data[i + 1]) == prev_l):
                check_ok = check_ok + 1
        if (check_ok == (len(raw_data) - 1)):
            return prev_l
        else:
            return -1

    def create_peaks(self, ds, size):
        vs = np.zeros(size)
        i = 0
        for i in range(len(ds)):
            vs[ds[i]] = 1
        return vs

    def plot_data(self, ds, raw_data, start=0, stop=0):
        data_size = len(raw_data)
        if (stop == 0):
            stop = len(raw_data)
        if (start > stop):
            print("stop setting error")
        x = np.arange(0, len(raw_data), 1)
        vs = self.create_peaks(ds, data_size)
        min_val = np.min(raw_data)
        mag = np.max(raw_data) - min_val

        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(6)
        plt.plot(x[start:stop],
                 vs[start:stop] * mag + min_val,
                 alpha=0.3,
                 color='r')
        plt.plot(x[start:stop], raw_data[start:stop], color='gray', alpha=0.7)
        plt.grid()
        plt.savefig("foo.png")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    print(f'Run signal sensor')
    print(f"Use the following config to produce results: {args.config}.")
    cfg = load_json(args.config)
    sensor = Sensor(cfg)
    sensor()
