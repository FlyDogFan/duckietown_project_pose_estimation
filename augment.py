from matplotlib import pyplot as plt
import numpy as np
import cv2
from kmeans import nn
from random_textures import sample

KEY = {"sky":[2], "road":[1], "ground":[3], "lines":[0]}
CC = np.load('./cc.npz')['cc']

def augment(im):
    pix = im.reshape((-1,3))
    imc = nn(pix,CC).reshape(im.shape[:-1])
    out = np.zeros_like(im, dtype=np.float)
    for k in KEY:
        mask = sum([imc == i for i in KEY[k]])
        out += mask[:,:,None] * sample(imc.shape) / 255.
    return out

if __name__=='__main__':
    from dataset import ImageFolder
    X = ImageFolder('../gym-duckietown/images')
    for i in range(20):
        im = X[i][0].numpy().transpose(1,2,0)
        a = augment(im)
        cv2.imshow('img', np.uint8(a * 255))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
