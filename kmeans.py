import numpy as np

_GRP = 1

def squared_dist(X1,X2=None):
    if X2 is None:
        G = X1.dot(X1.T)
        Z = np.diag(G)[None]
        return Z + Z.T - 2 * G
    else:
        G = X1.dot(X2.T)
        Z1 = np.sum(X1 ** 2, axis=1)[None]
        Z2 = np.sum(X2 ** 2, axis=1)[None]
        return  Z1.T + Z2 - 2 * G


def nn(X,cc,batch=1):
    cid = np.zeros((X.shape[0]),dtype=np.int32)
    for g in range(batch):
        d = squared_dist(X[g::batch], cc)
        cid[g::batch] = np.argmin(d,axis=1)
    return cid

def kmeans(X, n_clusters=10, iters=50, cfac=-1, batch=1):
    # Random init
    cc = np.random.permutation(X.shape[0])[:n_clusters]
    cc = X[cc]
    cc[0] = [i / 255. for i in [168, 139, 22]]
    cc[1] = [i / 255. for i in [26, 26, 26]]
    cc[2] = [i / 255. for i in [178, 178, 178]]
    cc[3] = [i / 255. for i in [243, 243, 243]]

    if cfac > 0:
        X2 = np.random.permutation(X.shape[0])[:(n_clusters * cfac)]
        X2 = X[X2]
    else:
        X2 = X

    # Do k-means
    for it in range(iters):
        cid = nn(X2,cc,batch)
        act=0
        for q in range(n_clusters):
            if np.any(cid==q):
                cc[q] = np.mean(X2[cid==q],axis=0)
                act=act+1
        print("It %03d, %05d active"%(it,act))

    cid = nn(X,cc)
    return cid, cc

def main():
    import dataset
    X = dataset.ImageFolder('./images')
    ind = np.random.permutation(len(X))[:500]
    X2 = []
    for i in ind:
        X2.append(X[i][0].numpy())
    X2 = np.array(X2).transpose(1,0,2,3).reshape(3,-1).T
    cid, cc = kmeans(X2, 4, 50, -1, 1)
    print(cid.shape)
    print(cc)
    out = {'cc':cc}
    np.savez('cc.npz', **out)


if __name__=='__main__':
    main()
