import cv2
import numpy as np
import os
from utils.segment import Segment
from utils.gmm import  fit_gmm 
from utils.k_means import fit_kmeans
from utils.utils import processing_normalization, get_diameter_distribution
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-transform_string', type=str , default="dft", help='')
    parser.add_argument('-stats', type=str , default="v", help='')
    parser.add_argument('-win_fac', type=float , default=0.03, help='')
    parser.add_argument('-pca', type=bool , default=False, help='')
    parser.add_argument('-pca_n_comp', type=int , default=3, help='')
    parser.add_argument('-hist_eql', type=bool , default=False, help='')
    parser.add_argument('-pix_to_nm', type=float , default=5.21, help='')
    parser.add_argument('-inp_path', type=str , default="./input/", help='')
    parser.add_argument('-clust_algo', type=str , choices=["gmm", "k_mean"], default="k_means", help='')
    parser.add_argument('-save', type=str , choices=["B", "B_DD"], default="k_means", help='')
    parser.add_argument('-out_path', type=str , default="./output/", help='')

    segment = Segment(pca_comp = 3,
                    pca_apply=False,
                    win_ratio = 0.03,
                    transform_string = "dft_dlresnet50", #"dl_resnet50",
                    stand_out_features = True,
                    eql_hist = False,
                    stats_to_use = "vas")
    um_pix=5.21
    inp_img = np.random.randint(low=0, high=254, size=(60,60), dtype=int)
    feature_cube_flat, w, h = segment.slinding_window(inp_img)
    feature_cube_flat = np.array(feature_cube_flat)
    
    feature_cube_flat_norm = processing_normalization(feature_cube_flat)
    
    index_map = fit_kmeans(feature_cube_flat_norm,w,h,no_clusters)
    
    index_map = index_map*255
    
    fig = get_diameter_distribution(index_map, um_pix)
