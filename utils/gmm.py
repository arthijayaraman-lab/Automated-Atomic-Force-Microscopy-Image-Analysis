import numpy as np 
from sklearn.mixture import GaussianMixture

def fit_gmm(feature_cube_flt, w, h, no_clusters):
    """
    predict clusters from feature cube using gmm
    Input:
        feature_cube_flt(ndarray) : feature cube flattened  
        w(int) : width of feature cube
        h(int) : height of feature cube
    return
        index_map(ndarray) : index map 
    """
    gmm = GaussianMixture(n_components=no_clusters)
    gmm.fit(dataset)
    ind_out = gmm.predict(dataset)
    out_img = np.zeros((w*h), dtype = float)
    for i in range(ind_out.shape[0]):
        out_img[i]=ind_out[i]
    index_map = out_img.reshape(w,h)
    return index_map
