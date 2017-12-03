import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import ImageFolder
from models import CNNPolicy

import random

model = CNNPolicy(3)
model.load_state_dict(torch.load('checkpoint.pth.tar')['state_dict'])
model.eval()
# model.load_state_dict(torch.load('model_best.pth.tar')['state_dict'])


random.seed(123)
idxs = list(range(10100))
random.shuffle(idxs)

data_set = ImageFolder('../gym-duckietown/images', return_path=True)
test_loader = torch.utils.data.DataLoader(
    data_set, batch_size=1, shuffle=False,
    num_workers=1, sampler=idxs[10000:10100])

output_html = ''

losses = []

for i, (input, target, path) in enumerate(test_loader):

    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)

    # compute output
    output = model(input_var)
    loss = ((output - target_var) ** 2).sum(1).mean()
    output_html += '<img src=%s>' %(path)

    output_html += '<p>%s, %s<br>%s, %s</p>' %(target[0,0],target[0,1], output.data[0,0],output.data[0,1])
    output_html += '<p>%s</p>' %(loss.data[0])

    losses.append(loss.data[0])

with open('index.html', 'w') as f:
    f.write(output_html)

print(sum(losses)/len(losses))