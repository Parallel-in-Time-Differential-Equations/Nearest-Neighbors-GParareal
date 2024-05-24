# Nearest Neighbors GParareal: Improving Scalability Of Gaussian Processes For Parallel In Time Solvers

This repository is the official implementation of [Nearest Neighbors GParareal: Improving Scalability Of Gaussian Processes For Parallel In Time Solvers](https://arxiv.org/abs/2405.12182). Authors: Guglielmo Gattiglio, Lyudmila Grigoryeva, and Massimiliano Tamborrino. 

>In this work we introduce nearest neighbors (nn) GParareal (nnGParareal), a novel data-enriched parallel-in-time (PinT) integration algorithm. nnGParareal builds upon GParareal, another PinT, by improving its scalability properties for higher-dimensional systems and increased processor count. This translates into sensibly faster parallel speed-up gains, compared to both Parareal and GParareal, over a range of systems
<!-- ![Non-linear Hopf Bifurcation](img/nonaut_scal_speedup.png)![FHN PDE](img/fhn_pde_speedup_upd.png) -->

 <p align="center">
  <img  src="img/nonaut_scal_speedup.png" width="48%" title="Non-linear Hopf Bifurcation system">
&nbsp; &nbsp; 
  <img  src="img/fhn_pde_speedup_upd.png" width="48%" title="FitzHugh-Nagumo Equations (PDE)">
  <!-- <p align="center"> Speed-up analysis of Non-linear Hopf Bifurcation system (left) and FitzHugh-Nagumo Equations (right). $a+ \hat{b}$</p> -->
</p>

Speed-up analysis of Non-linear Hopf Bifurcation system (left) and FitzHugh-Nagumo Equations (right). Legend: upper bound $S_{\rm alg}^*$, theoretical $S_{\rm alg}$ and empirical $\hat S_{\rm alg}$ speed-up values for alg = Para, GPara and nnGPara, standing for Parareal, GParareal, and nnGParareal, respectively.


## Requirements

The code was run using python 3.10. To install requirements:

```setup
pip install -r requirements.txt
```
Note: on some machines, `mpi4py` may fail to install. This is not necessary to reproduce the results when using the outputs provided. However, the reader interested in reproducing everything from scratch, including simulation outputs, will require MPI for parallelization and, ideally, at least 512 compute cores. WARNING: reproducing all the simulations from scratch will take several days.

For this reason, `mpi4py==3.1.5` has been commented in `requirements.txt` and _will not_ be installed by default.

Make sure you also have installed Python _pip_ and _venv_. If not, you can use (some of) the commands below
```
sudo apt install python3-pip
pip install virtualenv 

apt install python3.10-venv
```



## Results

The code and outputs to fully replicate the tables and figures in the paper are provided. Each file will usually contain both the code to run the experiment, and the analysis. They are orgained as follows:
- `Figure_*.py` or `Table_*.py`: code for the corresponding Figure or Table. 
- `Burgers.py`, `Hopf.py`, `TomLab.py`, `FHN_PDE.py`: code for running the corresponding scalability analysis. Additional results are provided in `Burgers_perf_across_m.py`, where the effect of $m$ on $K$ for Burgers is explored.
- `dataset_visualization.py`: code to illustrate the dataset used by nnGParareal (nearest neighbors) and the relative distance between observations.
- `nnGPara_with_time.py`: everything concerning to the "time" extension of nnGParareal.
- `Precision_plots_all_systems.py`: produces Figures 7 to 12.

Other files:
- `preprocessing.py`: some simulation outputs have been processed to make them lighter, by removing unnecessary information.


## How To Use

To replicate a specific analysis, say `Figure_3.py`, run the following commands on a Linux shell. A similar strategy can be followed for Windows, but the commands need to be changed. Alternatively, install Windows Subsystem for Linux.
```bash
wget https://github.com/Parallel-in-Time-Differential-Equations/Nearest-Neighbors-GParareal/archive/refs/heads/main.zip
tar -xzf main.tar.gz
cd Nearest-Neighbors-GParareal-main/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 Figure_3.py
```
Some users may need to replace the last line with `python Figure_3.py`, based on their platform. Figure_3's code does not produce any textual output, but you can find Figure 3 in `img/nngp_m_distr_oneline.pdf`.

## Acknowledgments

Part of the code has been translated and adapted from [GParareal](https://github.com/kpentland/GParareal)'s Matlab implementation .

