import numpy as np
import cv2

def generate_noise(shape):
    return np.random.uniform(0,1,shape)

def zoom(X, frac=.5):
    ns = [int(s * frac) for s in X.shape]
    return cv2.resize(X[:ns[0],:ns[1]], X.shape)

def turbulence(X, n=2):
    Y = np.zeros_like(X)
    for i in range(n):
        Y += zoom(X,1./(2**i)) * (2 ** i)
    Y /= sum([2 ** i for i in range(n)])
    return Y

def marble(X, angle=np.pi/4, freq=5., power=5.):
    x = np.ones_like(X)
    x = np.cumsum(x,axis=0)
    x /= np.max(x)
    xysum = np.cos(angle) * x + np.sin(angle) * x.T
    return (np.sin(2 * np.pi * (freq * xysum + power * X)) + 1) / 2.

def cross(X, freq=5., power=5.):
    x = np.ones_like(X)
    x = np.cumsum(x,axis=0)
    x /= np.max(x)
    xval = (np.sin(np.pi * (freq * x + power * X)) + 1) / 2.
    yval = (np.sin(np.pi * (freq * x.T + power * X[::-1,::-1])) + 1) / 2.
    return (xval + yval) / 2

def wood(X, freq=5., power=5.):
    x = np.ones_like(X)
    x = np.cumsum(x,axis=0)
    x = (x / np.max(x)) - .5
    dist = np.sqrt(x ** 2 + x.T ** 2)
    return (np.sin(2 * np.pi * (freq * dist + power * X)) + 1) / 2.


def sample_marble(shape):
    X = turbulence(generate_noise(shape), n=4)
    angle = np.random.uniform(0,2 * np.pi)
    freq = np.random.uniform(1,10)
    power = np.random.uniform(2,5)
    X = marble(X, angle, freq, power)[:,:,None]
    c1 = sample_color()
    c2 = sample_color()
    return X * c1 + (1. - X) * c2

def sample_cross(shape):
    x = turbulence(generate_noise(shape), n=4)
    freq = np.random.uniform(1,10)
    power = np.random.uniform(2,5)
    X = cross(x, freq, power)[:,:,None]
    c1 = sample_color()
    c2 = sample_color()
    return X * c1 + (1. - X) * c2

def sample_wood(shape):
    x = turbulence(generate_noise(shape), n=4)
    freq = np.random.uniform(1,10)
    power = np.random.uniform(1,4)
    X = wood(x, freq, power)[:,:,None]
    c1 = sample_color()
    c2 = sample_color()
    return X * c1 + (1. - X) * c2

def sample_clouds(shape):
    n = np.random.randint(2,4)
    X = turbulence(generate_noise(shape), n=n)[:,:,None]
    c1 = sample_color()
    c2 = sample_color()
    return X * c1 + (1. - X) * c2

def sample_solid(shape):
    pattern = np.ones(shape)
    color = sample_color()
    return pattern[:,:,None] * color

def sample_color():
    return np.random.randint(0,256,size=[1,1,3])

def sample(shape):
    types = {0:sample_solid, 1:sample_clouds, 2:sample_wood, 3:sample_cross, 4:sample_marble}
    return types[np.random.randint(0,5)](shape)


if __name__=='__main__':
    for i in range(1,20):
        im = np.uint8(sample((64,64)))
        cv2.imshow('img',im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
