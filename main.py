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
    parser.add_argument('-T', metavar='Transforms', type=str , default="dft", help='transform list passed as a string ')
    parser.add_argument('-S', metavar='Statistics', type=str , default="v", help='List of statistics used in dimensional reduction and feature extraction. a = mean; m = max; s = skew; k = kurtosis; v = variance (e.g. mean+variance => python3 main.py -S "av")')
    parser.add_argument('-wf', metavar='Win_factor', type=float , default=0.03, help='Win_factor controls tile size is the ratio of tile size to input image')
    parser.add_argument('-pca', action='store_true', help='Flag to use PCA')
    parser.add_argument('-npcs', metavar='no_of_pca', type=int , default=3, help='Number of components to extract in PCA')
    parser.add_argument('-hist_eql', action='store_true', help='Flag to perform histogram equalization on pre-processed image')
    parser.add_argument('-pix_to_nm', metavar='pixel_to_nanometer', type=float , default=5.21, help='Scale to convert one pixel into real distance units (nm/pix in our case)')
    parser.add_argument('-i', metavar='Input_dir', type=str , default="./input/", help='Folder contain input images to process')
    parser.add_argument('-C', metavar='Clustering_algo', type=str , choices=["gmm", "k_mean"], default="k_mean", help='Choose which clustering algorithum to use')
    parser.add_argument('-NC', type=int , default=2, help='Number of clusters the clustering algorithm needs to generate')
    parser.add_argument('-W', type=str , choices=["B", "B_DD"], default="k_means", help='writing options: decides all the file that are to be output')
    parser.add_argument('-O', type=str , default="./output/", help='Output destination path')
    args = parser.parse_args()
    
    segment = Segment(pca_comp = args.pca_n_comp,
                    pca_apply=args.pca,
                    win_ratio = args.win_fac,
                    transform_string = args.transform_string, #"dl_resnet50",
                    stand_out_features = True,
                    eql_hist = args.hist_eql,
                    stats_to_use = args.stats)

    inp_img = np.random.randint(low=0, high=2054, size=(60,60), dtype=int)
    feature_cube_flat, w, h = segment.slinding_window(inp_img)
    feature_cube_flat = np.array(feature_cube_flat)
    
    feature_cube_flat_norm = processing_normalization(feature_cube_flat)
    
    if args.clust_algo == "k_mean":
        index_map = fit_kmeans(feature_cube_flat_norm,w,h,args.n_clst)
    elif args.clust_algo == "gmm":
        index_map = fit_gmm(feature_cube_flat_norm,w,h,args.n_clst)
        
    index_map = index_map*255
    
    fig = get_diameter_distribution(index_map, args.pix_to_nm)
    print(index_map)
