
# Automated Atomic Force Microscopy Image Analysis

An unsupervised machine learning based workflow that automates identification and quantification of features (domain sizes example shows in repo) in AFM images of polymer films.





## Overview

![](https://github.com/arthijayaraman-lab/Automated-Atomic-Force-Microscopy-Image-Analysis/blob/main/img/overview_flow.gif)
 
![](https://github.com/arthijayaraman-lab/Automated-Atomic-Force-Microscopy-Image-Analysis/blob/main/img/Demo.gif)


## Features

- Unsupervised solution - does not require model training 
- No training labels required 
- Generalizable to other polymer films 
 



## Documentation

 The project is an implementation of the paper \
 ["Machine Learning for Analyzing Atomic Force Microscopy
(AFM)
Images
Generated from
Polymer Blends"]([https://linktopaper](https://arxiv.org/abs/2409.11438))



## Workflow

![workflow](https://github.com/arthijayaraman-lab/Automated-Atomic-Force-Microscopy-Image-Analysis/blob/main/img/workflow.png)


## Dataset

![Data_preview](https://github.com/arthijayaraman-lab/Automated-Atomic-Force-Microscopy-Image-Analysis/blob/main/img/dataset_preview.png)

**Dataset avilable at [zenodo](https://zenodo.org/records/11179874?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjM0MWViYzY4LTI1NzMtNGYxMC1iZjdjLTUwYWQ4Zjk0NGU0OSIsImRhdGEiOnt9LCJyYW5kb20iOiJjMDcwMTc3Y2IxNzM3ZGMxZWU1MWU2MjJjMjA0N2ZjMCJ9.Y3_qzNSNsap_oqLRnEi-wHmwooy65TT6F7tjFTF5qE0X8evYr0VTZmGKh34TI6UmsAd9cJrfnlbm6rQUK82h7A)**

#### Details
- Total **144** images of **16** polymer samples
- **9** images of each sample 
- raw **`.ibw`** files 
- **384** x **384** pixels image size  




## How to use

There are two ways to run the code. 
- Jupyter Notebook (**Recommended - Simple**)

- Python3 terminal

### Jupyter Notebook  
- Download [python notebook](https://github.com/arthijayaraman-lab/Automated-Atomic-Force-Microscopy-Image-Analysis/blob/main/notebooks/auto_afm.ipynb)
 - (**Prefered**) upload to https://colab.research.google.com/  
        OR 
- Upload locally into jupyter notebook 

### Python3 Terminal  
#### Installation

Download code from Github


```bash 
git clone https://github.com/arthijayaraman-lab/Automated-Atomic-Force-Microscopy-Image-Analysis.git
cd Automated-Atomic-Force-Microscopy-Image-Analysis
```

Install conda environment

```bash
conda env create -f environment.yml
```
**Run code**\
Run main script main.py
```bash 
python3 main.py 

```
Help options `main.py`   
```bash
python3 main.py -h
```
Arguments to `main.py`
```bash
usage: main.py [-h] [-T Transforms] [-S Statistics] [-wf Win_factor] [-pca] [-npcs no_of_pca] [-hist_eql] [-pix_to_nm pixel_to_nanometer] [-i Input_dir] [-C Clustering_algo] [-NC NC]
               [-W {B,B_DD}] [-O O]

options:

  -h, --help            show this help message and exit
  -T Transforms         transform list passed as a string
  -S Statistics         List of statistics used in dimensional reduction and feature extraction. a = mean; m = max; s = skew; k = kurtosis; v = variance (e.g. mean+variance => python3 main.py -S
                        "av")
  -wf Win_factor        Win_factor controls tile size is the ratio of tile size to input image
  -pca                  Flag to use PCA
  -npcs no_of_pca       Number of components to extract in PCA
  -hist_eql             Flag to perform histogram equalization on pre-processed image
  -pix_to_nm pixel_to_nanometer
                        Scale to convert one pixel into real distance units (nm/pix in our case)
  -i Input_dir          Folder contain input images to process
  -C Clustering_algo    Choose which clustering algorithm to use
  -NC NC                Number of clusters the clustering algorithm needs to generate
  -W {B,B_DD}           writing options: decides all the file that are to be output
  -O O                  Output destination path


```
#### Example 
```
python3 main.py -T "dft" -S "v" -wf 0.03 -pix_to_nm 5.21 -i /path/to/input/dir/ -C "k_mean" -NC 2 -O /path/to/output/dir/
```
**Note**:- currently only supports **`.png`** files as input. 




## Acknowledgements
A.J. and A.P. are grateful for financial support from Multi University Research Initiative (MURI) from the Army Research Office, Award Number W911NF2310260.. Y. W. and X. G. are grateful for financial support from the Department of Energy under the award number of DE-SC0024432. A portion of this work was done at the Molecular Foundry, which is supported by the Office of Science, Office of Basic Energy Sciences, of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231.


## Citation
Cite us,
```bibtex
 @article{paruchuri2024machine,
  title={Machine Learning for Analyzing Atomic Force Microscopy (AFM) Images Generated from Polymer Blends},
  author={Paruchuri, Aanish and Wang, Yunfei and Gu, Xiaodan and Jayaraman, Arthi},
  journal={arXiv preprint arXiv:2409.11438},
  year={2024}
}
```
