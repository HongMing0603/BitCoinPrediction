import numpy as np
def mape(x1, x2, axis=0):
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)

    return np.mean(abs((x1-x2)/x1),axis=axis)*100


def smape(a,f):

    a= np.asanyarray(a)
    f=np.asanyarray(f)

    return 1/len(a) * np.sum(2*np.abs(f-a)/(np.abs(a)+np.abs(f))*100)