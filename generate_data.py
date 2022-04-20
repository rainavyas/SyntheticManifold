'''
Generates test and train data
3D-Data points are generated from class A and class B, each with their Gaussians with their
own mean (only along x-axis) and a fixed covariance matrix with diagonals variance set to s2.
Variance of z-axis is set to zero, so points are generated on x-y plane only.

Data files will be saved as .npy files with structure: [num_points x 4],
where the 4 columns are: [x, y, z, class], and class = {0,1}
'''

import sys
import os
import argparse
import numpy as np
import random

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('OUT', type=str, help='Base path to save generated data')
    commandLineParser.add_argument('--mean_shift', type=float, default=1, help='Gaussian mean shifts from x-axis')
    commandLineParser.add_argument('--var', type=float, default=1, help='Gaussian diag variance')
    commandLineParser.add_argument('--seed', type=int, default=1, help='reproducibility')
    commandLineParser.add_argument('--train_points_per_class', type=int, default=1000, help='reproducibility')
    commandLineParser.add_argument('--test_points_per_class', type=int, default=1000, help='reproducibility')
    commandLineParser.add_argument('--save_test', type=str, default='no', help='do you want to re-save test data')

    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/generate_data.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    set_seeds(args.seed)

    # Generate data points from class A and split
    mean = [-1*args.mean_shift, 0, 0]
    cov = [[args.var, 0, 0], [0, args.var, 0], [0, 0, 0]]
    num_points = args.train_points_per_class + args.test_points_per_class
    data = np.random.default_rng().multivariate_normal(mean, cov, num_points)
    class_label = np.zeros((num_points, 1))
    data = np.hstack((data, class_label))
    data_testA = data[:args.test_points_per_class]
    data_trainA = data[args.test_points_per_class:]

    # Generate data points from class B and split
    mean = [args.mean_shift, 0, 0]
    cov = [[args.var, 0, 0], [0, args.var, 0], [0, 0, 0]]
    num_points = args.train_points_per_class + args.test_points_per_class
    data = np.random.default_rng().multivariate_normal(mean, cov, num_points)
    class_label = np.ones((num_points, 1))
    data = np.hstack((data, class_label))
    data_testB = data[:args.test_points_per_class]
    data_trainB = data[args.test_points_per_class:]

    # Merge A and B classes
    data_train = np.vstack((data_trainA, data_trainB))
    data_test = np.vstack((data_testA, data_testB))
    # import pdb; pdb.set_trace()

    # Save the data
    np.save(f'{args.OUT}/train{args.train_points_per_class}_mean_shift{args.mean_shift}_var{args.var}.npy')
    if args.save_test == 'yes':
        np.save(f'{args.OUT}/test{args.test_points_per_class}_mean_shift{args.mean_shift}_var{args.var}.npy')


