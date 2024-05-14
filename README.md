
# Automated Atomic Force Microscopy Image Analysis

An unsupervised machine learning based workflow that automates identification and quantification of features (domain sizes exmaple showns in repo) in AFM images of polymer blends.




## Overview

![](https://github.com/arthijayaraman-lab/Automated-Atomic-Force-Microscopy-Image-Analysis/blob/main/img/Demo.gif)

![](https://github.com/arthijayaraman-lab/Automated-Atomic-Force-Microscopy-Image-Analysis/blob/main/img/overview_flow.gif)
 
## Features

- Unsupervised solution - does not require model training 
- No training lables required 
- Genralizable to different polymer blends 
 


## Documentation

 The project is an implemnetation of the paper \
 ["Machine Learning for Analyzing Atomic Force Microscopy
(AFM)
Images
Generated from
Polymer Blends"](https://linktopaper)


## Workflow

![workflow](https://github.com/arthijayaraman-lab/Automated-Atomic-Force-Microscopy-Image-Analysis/blob/main/img/workflow.png)


## Dataset

![Data_preview](https://github.com/arthijayaraman-lab/Automated-Atomic-Force-Microscopy-Image-Analysis/blob/main/img/dataset_preview.png)

**Dataset avilable at [zenodo](https://zenodo.org/records/11179874?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjM0MWViYzY4LTI1NzMtNGYxMC1iZjdjLTUwYWQ4Zjk0NGU0OSIsImRhdGEiOnt9LCJyYW5kb20iOiJjMDcwMTc3Y2IxNzM3ZGMxZWU1MWU2MjJjMjA0N2ZjMCJ9.Y3_qzNSNsap_oqLRnEi-wHmwooy65TT6F7tjFTF5qE0X8evYr0VTZmGKh34TI6UmsAd9cJrfnlbm6rQUK82h7A)**

#### Deatils
- Total **144** images 
- **16** unique block copolymer blends of varing chain length and chain ratio
- **9** measurements of each blend 
- raw **`.ibw`** files 
- **384** x **384** pixels image size  

## How to use

There are two ways to run the code. 
- Jupyter Notebook (**Recomended - Simple**)

- Python3 terminal

### Jupyter Notebook  
- Download [python notebook](https://github.com/arthijayaraman-lab/Automated-Atomic-Force-Microscopy-Image-Analysis/blob/main/notebooks/auto_afm.ipynb)
 - (**Prefered**) upload to https://colab.research.google.com/  
        OR 
- Upload locally into jupyter notebook 

### Python3 Terminal  
#### Installation

Download code from repo


```bash 
  git clone https://github.com/arthijayaraman-lab/Automated-Atomic-Force-Microscopy-Image-Analysis.git
  cd Automated-Atomic-Force-Microscopy-Image-Analysis
```

Install conda enveronment

```bash
conda env create -f environment.yml
```
**Run code**\
Run main script main.py
```bash 
python3 main.py 

```


## Other Use Cases
first show our case code to run 

second show liturature data how it runs with dwt 

third show feature extraction from lit data on fiberals 
## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


## Citation
Cite us,
```bibtex
    citation 
```