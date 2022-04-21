'''
Perform FGSM or PGD attack on data points
Save the attacked data points to a file
'''

import torch
import torch.nn as nn
import sys
import os
import argparse
from models import FFN
import numpy as np

def fgsm_attack(x, y, model, criterion, epsilon):
    '''
    x: torch.Tensor [Bx3]
    y: torch.Tensor [B]
    
    Perform Finite Gradient Sign Method attack
    '''
    model.eval()
    s = nn.Sigmoid()
    x.retain_grad()
    pred = s(model(x))
    loss = criterion(pred, y)
    loss.backward()

    x_attacked = x + epsilon*torch.sign(x.grad)
    import pdb; pdb.set_trace()
    return x_attacked


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='Specify model th file path')
    commandLineParser.add_argument('ARCH', type=str, help='ffn')
    commandLineParser.add_argument('ATTACK', type=str, help='fgsm')
    commandLineParser.add_argument('ORIG_DATA', type=str, help='path to test.npy file')
    commandLineParser.add_argument('OUT_DATA', type=str, help='path to dir to save attacked test.npy file')
    commandLineParser.add_argument('--num_hidden_layers', type=int, default=1, help="number of hidden layers")
    commandLineParser.add_argument('--hidden_layer_size', type=int, default=10, help="size of hidden layers")
    commandLineParser.add_argument('--epsilon', type=float, default=0.1, help="attack size")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load the data as tensors
    with open(args.ORIG_DATA, 'rb') as f:
        data = np.load(f)
    data = torch.from_numpy(data)
    x = data[:,:-1].type(torch.FloatTensor)
    y = data[:,-1].type(torch.FloatTensor)

    # Initialise classifier
    model = FFN(num_hidden_layers=args.num_hidden_layers, hidden_layer_size=args.hidden_layer_size)
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))

    # Criterion
    criterion = nn.BCELoss()

    # Attack
    if args.ATTACK == 'fgsm':
        x_attacked = fgsm_attack(x, y, model, criterion, args.epsilon)
    
    # Save the attacked data
    np.save(f'{args.OUT}/{args.ORIG_DATA[:-4]}_{args.ATTACK}_epsilon{args.epsilon}.npy', x_attacked.cpu().detach().numpy())