import cv2
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
import pywt
import scipy.ndimage as spim
from skimage.transform import radon
from sklearn.decomposition import PCA
from scipy.fftpack import dct
import time
from utils.utils import processing_normalization, get_transform_list, get_stats_feature_vector

class Segment:
  def __init__(self, pca_comp = 3, pca_apply=False, win_ratio = 0.03, transform_string = "dft", stand_out_features = True, eql_hist = False, stats_to_use = "va"):
    self.pca_comp = pca_comp    # No. of Principal components to decompose 
    self.pca_apply = pca_apply  # flag to apply Principal component analysis on features
    self.win_ratio = win_ratio  # win factor (decides the size of the tile)
    self.transform_string = transform_string  # list of tranforms applied to extract features
    self.stand_out_features = stand_out_features  # flag to normalize features
    self.eql_hist = eql_hist  # flag to perform histogram equalization of input image 
    self.stats_to_use = stats_to_use  # list of satistics to use
    self.model_stack = []  
    self.layer_names = ['conv1_relu', 'conv2_block3_2_relu', 'conv3_block1_2_relu']  # layers to use from ResNet50 for feature extraction
    self.transform_list = get_transform_list(self.transform_string)

      
  ############################################
  ############# Loop functions ###############
  ############################################

  def get_features(self, img1, transform):
    """
    function to extract features from tile.
    Input:
      img1(ndarray) : tile  
      transform(list (str)) : list of transform to perform to extract features  
    return:
      all_features(ndarray)  : feature vector
    """
    if "dlresnet50" in transform:
      self.load_ResNet50()
      all_features = self.get_ResNet50(img1)
    else:
      all_features = np.array([])
    if "radon" in transform :
      all_features=np.append(all_features,self.get_radon_features(img1), axis=0)
    if "dft" in transform :
      all_features=np.append(all_features,self.get_dft_features(img1), axis=0)
    if "dct" in transform :
      all_features=np.append(all_features,self.get_dct_features(img1), axis=0)
    if "dwt_biort" in transform :
      all_features=np.append(all_features,self.get_biort_features(img1), axis=0)
    if "dwt_haar" in transform :
      all_features=np.append(all_features,self.get_wavelget_haar_featureset_features(img1), axis=0)
    return all_features

  def slinding_window(self, img):
    """
    function to make tiles
    features:
      input(ndarray) : input image 
    return:
      all_trans_features(ndarray) : feature cube flattened
      i_ctr(int) : feature cube width
      j_ctr(int) : feature cube height
    """
    for transform in self.transform_list:
      img_features = []
      if "dlresnet50" in self.transform_list:
        win_siz = 32
        img_tiles = []
      else:
        win_siz = round((img.shape[0]*self.win_ratio + img.shape[1]*self.win_ratio)/2)

      i_ctr = 0
      for i in range(win_siz//2,img.shape[0]-(win_siz//2),1):
        j_ctr = 0
        for j in range(win_siz//2,img.shape[1]-(win_siz//2),1):
          if "dlresnet50" in transform:
            img_tiles.append(img[(i-(win_siz//2)):(i+(win_siz//2)), (j-(win_siz//2)):(j+(win_siz//2))])
          else:
            img_features.append(self.get_features(img[(i-(win_siz//2)):(i+(win_siz//2)), (j-(win_siz//2)):(j+(win_siz//2))],transform))
          j_ctr+=1
        i_ctr+=1
      if "dlresnet50" in transform:
        img_features = self.get_features(img_tiles, transform)
      img_features = np.array(img_features)
      if "all_trans_features" not in locals():
        all_trans_features = img_features.copy()
      else:
        all_trans_features = np.append(all_trans_features, img_features, axis=1)

    return all_trans_features, i_ctr, j_ctr

  ############################################
  ############## DL ResNet50 #################
  ############################################

  def load_ResNet50(self):
    """
    function to initalize ResNet50 model 
    Return:
      tf_model_object, tf_model_object, ...
    """
    model = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(32,32,3),
        pooling=None
    )

    for lyr in self.layer_names:
      self.model_stack.append(Model(inputs=model.input, outputs=model.get_layer(lyr).output))
    return self.model_stack

  def get_ResNet50(self, img):
    """
    function that returns features generated from ResNet50 model.  
    Input:
        img(ndarray) : tile 
    return:
        out_features(ndarray) : feature vector 

    """
    img = np.array(img)
    img = np.repeat(img[:, :, :, np.newaxis], 3, axis=3) # repeat gray scale image into 3 channels like rgb

    for img_t in np.split(img, 8, axis=0):  #maxPrimeFactors(img.shape[0])
      if len(self.model_stack)>0:
        output_1 = self.model_stack[0].predict(img_t)
        features = np.max(output_1,axis=(1, 2))
        features = np.append(features, np.mean(output_1,axis=(1, 2)), axis =1)
        features = np.append(features, np.var(output_1,axis=(1, 2)), axis =1)
      if len(self.model_stack)>1:
        output_2 = self.model_stack[1].predict(img_t)
        features = np.append(features, np.max(output_2,axis=(1, 2)), axis =1)
        features = np.append(features, np.mean(output_2,axis=(1, 2)), axis =1)
        features = np.append(features, np.var(output_2,axis=(1, 2)), axis =1)
      if len(self.model_stack)>2:
        output_3 = self.model_stack[2].predict(img_t)
        features = np.append(features, np.max(output_3,axis=(1, 2)), axis =1)
        features = np.append(features, np.mean(output_3,axis=(1, 2)), axis =1)
        features = np.append(features, np.var(output_3,axis=(1, 2)), axis =1)
      if 'out_features' not in locals():
        out_features = features.copy()
      else:
        out_features = np.append(out_features,features, axis = 0)
    
    return out_features



  ############################################
  ########### Domain Transforms ##############
  ############################################

  ####### Discrete Cosine transforms ########
  def get_dct_features(self, tile):
    """
    Discrete cosine transform (DCT) based feature extraction
    Input:
        tile(ndarray) : tile 
    return
        self.get_stats_feature_vector(dct_image) (ndarray) : feature vectore 
    """
    dct_image = dct(dct(tile.T, norm='ortho').T, norm='ortho')
    return get_stats_feature_vector(dct_image, self.stats_to_use)

  ###### Discrete Fourirer transforms ########

  def get_dft_features(self, tile):
    """
    Discrete fourier transform (DFT) based feature extraction
    Input:
        tile(ndarray) : tile 
    return
        self.get_stats_feature_vector(dft_image) (ndarray) : feature vectore 
    """
    dft_image = np.fft.fft2(tile)
    dft_image = np.fft.fftshift(dft_image)
    return get_stats_feature_vector(dft_image, self.stats_to_use)


  ########## Wavelet transforms ##############
  def get_biort_features(self, tile, level=2):
    """
    Discrete wavelet transform (DWT) bi-orthogonal wavelet based feature extraction
    Input:
        tile(ndarray) : tile 
        level(int) : level of decomposition
    return
        texture_features(ndarray) : feature vectore 
    """

    coeffs = pywt.swt2(tile, 'bior5.5', start_level=0, level = level)
    cD = coeffs[:][1]
    texture_features = []
    for detail_coeff_ind in range(len(cD)):
        for i in range(3):
            texture_features.append(get_stats_feature_vector(cD[detail_coeff_ind][1][i]), self.stats_to_use)
    return texture_features

  def get_haar_features(self, tile, level = 4):
    """
    Discrete wavelet transform (DWT) Haar wavelet based feature extraction
    Input:
        tile(ndarray) : tile 
        level(int) : level of decomposition
    return
        texture_features(ndarray) : feature vectore 
    """
      
    coeffs = pywt.wavedec2(tile, "haar", level=level)
    cA, cD = coeffs[0], coeffs[1:]
    texture_features = []

    for detail_coeff in cD:
        texture_features.append(get_stats_feature_vector(detail_coeff), self.stats_to_use)

    return texture_features

  ########### Radon transform  ###############
  def get_radon_features(self, tile):
    """
    Radon transform based feature extraction
    Input:
        tile(ndarray) : tile 
    return
        texture_features(ndarray) : feature vectore 
    """
    theta = np.linspace(0., 180., max(tile.shape), endpoint=False)
    sinogram = radon(tile, theta=theta, circle=False)
    texture_features = get_stats_feature_vector(sinogram, self.stats_to_use)
    return texture_features



  ########## PCA ##########
  def processing_apply_pca(self, feature_cube_flt):
    """
    function to apply Principal Component Analysis
    Input:
        feature_cube_flt(ndarray) : feature cube flattened 
    output:
        principalComponents(ndarray) : principal components
    """
    feature_cube_flt = processing_normalization(feature_cube_flt)
    pca = PCA(n_components=self.pca_comp)
    principalComponents = pca.fit_transform(feature_cube_flt)
    return principalComponents





