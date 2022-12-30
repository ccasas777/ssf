from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import leastsq
import os
import argparse


def parse_txt(text_lines):
    time, ch1, ch2 = [], [], []
    for i, line in enumerate(text_lines):
        if i < 6:
            continue
        line = line.split("; ")
        try:
            time.append(float(line[0]))
            ch1.append(float(line[1]))
            ch2.append(float(line[2]))
        except:
            print("End line at {} ".format(i))
            break
    return np.asarray(time), np.asarray(ch1), np.asarray(ch2)


def load_text(path):
    with open(path) as f:
        return [l.replace("\n", "") for l in f.readlines()]


def auto_searching(data, window):
    temp_max = 0
    i = 0
    ind_min = 0
    ind_max = 0
    A = data[0:window]
    B = data[window+1:2*window+1]
    if((np.max(B)-np.max(A)) > 0):
        direct = "up"
    else:
        direct = "down"

    up_done = 0
    down_done = 0
    while(1):
        A = data[i:i+window]
        B = data[i+window+1:i+2*window+1]
        ################################
        #
        #  Case1 B == A : flat region
        #  Case2 B > A : go up
        #  Case3 B < A : go down
        #
        #############################
        # Case 1
        if ((np.max(B) - np.max(A)) == 0):
            i = i + window

        # Case 2
        if(direct == "up"):
            # start collect
            if((np.max(B) - np.max(A)) > 0):
                i = i + window
            else:
                ind_max = i + np.argmax(A)
                direct = "down"
                up_done = 1
        # Case 3
        if(direct == "down"):
            if((np.min(A) - np.min(B)) > 0):
                i = i+window
            else:
                ind_min = i+np.argmin(A)
                direct = "up"
                down_done = 1

        # Succesful auto searching
        if (down_done & up_done):
            return [ind_max, ind_min]


def triangle_wave(x, p):
    a, b, c, T = p
    y = np.where(np.mod(x-b, T) < T/2, -4/T*(np.mod(x-b, T))+1+c/a, 0)
    y = np.where(np.mod(x-b, T) >= T/2, 4/T*(np.mod(x-b, T))-3+c/a, y)
    return a*y


def residuals(p, y, x):
    return y - triangle_wave(x, p)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--Input", help='Enter Output path')

args = parser.parse_args()

if (args.Input):
    data_path = './raw_data/' + str(args.Input)

    print("loading data...")
    time, raw_data, triangle_waves = parse_txt(load_text(data_path))
    # auto searching parameters of the fitting
    pt_h0, pt_l0 = auto_searching(triangle_waves, 5000)
    mag = abs(triangle_waves[pt_h0] - triangle_waves[pt_l0])/2
    mid = triangle_waves[pt_l0] + mag
    T = abs(pt_h0 - pt_l0)*2

    # start fitting
    print("start fitting...")
    p0 = [mag, -pt_h0, mid, T]
    x = np.arange(0, len(triangle_waves), 1)
    plsq = leastsq(residuals, p0, args=(triangle_waves, x))

    up = np.arange(round(plsq[0][1]+plsq[0][3]),
                   len(triangle_waves), round(plsq[0][3]))
    down = up + round(plsq[0][3]/2)

    # start save
    print("saving result...")
    output_dir = './simulator/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    default_name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z']
    i = 0
    while(1):
        path = output_dir + default_name[i]

        if not os.path.exists(path):
            print('mkdir ' + path)
            os.mkdir(path)
            os.mkdir(path+'/down')
            os.mkdir(path+'/up')
            break
        else:
            i = i + 1
            if (i > 25):
                print("dir error")

    for i in range(len(up)-1):
        np.savetxt(path + '/down/'+str(i)+'.txt', raw_data[up[i]:down[i]])
        np.savetxt(path + '/up/'+str(i)+'.txt', raw_data[down[i]:up[i+1]])

    tmp = []
    for i in range(len(up)):
        tmp.append([up[i], down[i]])
    np.savetxt(path + '/index.txt', tmp)
