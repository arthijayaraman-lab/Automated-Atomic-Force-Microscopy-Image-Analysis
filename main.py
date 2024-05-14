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
    parser.add_argument('-T', '--transform_string', type=str , default="dft", help='')
    parser.add_argument('-S', '--stats', type=str , default="v", help='')
    parser.add_argument('-wf', '--win_fac', type=float , default=0.03, help='')
    parser.add_argument('-pcs', '--pca', type=bool , default=False, help='')
    parser.add_argument('-npcs', '--pca_n_comp', type=int , default=3, help='')
    parser.add_argument('-he', '--hist_eql', type=bool , default=False, help='')
    parser.add_argument('-ps', '--pix_to_nm', type=float , default=5.21, help='')
    parser.add_argument('-i', '--inp_path', type=str , default="./input/", help='')
    parser.add_argument('-C', '--clust_algo', type=str , choices=["gmm", "k_mean"], default="k_mean", help='')
    parser.add_argument('-NC', '--n_clst', type=int , default=2, help='')
    parser.add_argument('-W', '--save', type=str , choices=["B", "B_DD"], default="k_means", help='')
    parser.add_argument('-O', '--out_path', type=str , default="./output/", help='')
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
