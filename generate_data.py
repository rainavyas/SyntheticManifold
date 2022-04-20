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
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/generate_data.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    set_seeds(args.seed)

    # Generate data points from class A
    mean = [-1*args.mean_shift, 0, 0]
    cov = [[args.var, 0, 0], [0, args.var, 0], [0, 0, 0]]
    num_points = args.train_points_per_class + args.test_points_per_class
    data = np.random.default_rng().multivariate_normal(mean, cov, num_points)
    import pdb; pdb.set_trace()

