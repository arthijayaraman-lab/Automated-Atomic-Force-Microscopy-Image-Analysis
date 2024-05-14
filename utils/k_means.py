import numpy as np 
from sklearn.cluster import KMeans

def fit_kmeans(feature_cube_flt, w, h, no_clusters):
    """
    predict clusters from feature cube using k-means

    Input:
        feature_cube_flt(ndarray) : feature cube flattened  
        w(int) : width of feature cube
        h(int) : height of feature cube
    return
        index_map(ndarray) : index map 
    """
    kmeans = KMeans(n_clusters=no_clusters)
    kmeans.fit(feature_cube_flt)
    ind_out = kmeans.predict(feature_cube_flt)
    out_img = np.zeros((w*h), dtype = float)
    for i in range(ind_out.shape[0]):
        out_img[i]=ind_out[i]
    index_map = out_img.reshape(w,h)
    return index_map
