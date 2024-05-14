
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


## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


## Authors

- [@octokatherine](https://www.github.com/octokatherine)


## Citation
Cite us,
```bibtex
    citation 
```