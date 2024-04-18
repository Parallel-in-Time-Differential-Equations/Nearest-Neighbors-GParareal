# Nearest Neighbors Gparareal Improving Scalability Of Gaussian Processes For Parallel In Time Solvers

This repository is the official implementation of [Nearest Neighbors Gparareal Improving Scalability Of Gaussian Processes For Parallel In Time Solvers](). 

>In this work we introduce Nearest Neighbors GParareal (NN-GParareal), a novel data-enriched parallel-in-time (PinT) integration algorithm. NN-GParareal builds upon GParareal, another PinT, by improving its scalability properties for higher-dimensional systems and increased processor count. This translates into sensibly faster parallel speed-up gains, comapred to both Parareal and GParareal, over a range of systems
<!-- ![Non-linear Hopf Bifurcation](img/nonaut_scal_speedup.png)![FHN PDE](img/fhn_pde_speedup_upd.png) -->

<p align="center">
  <img alt="Light" src="img/nonaut_scal_speedup.png" width="48%">
&nbsp; &nbsp; 
  <img alt="Dark" src="img/fhn_pde_speedup_upd.png" width="48%">
</p>



## Requirements

The code was run using python 3.10. To install requirements:

```setup
pip install -r requirements.txt
```


## Results

The code and outputs to fully replicate the tables and figures in the paper are provided. Each file will usually contain both the code to run the experiment, and the analysis. They are orgained as follows:
- `Figure_*.py` or `Table_*.py`: code for the corresponding Figure or Table. 
- `Burgers.py`, `Hopf.py`, `TomLab.py`, `FHN_PDE.py`: code for running the corresponding scalability analysis. Additional results are provided in `Burgers_perf_across_m.py`, where the effect of $m$ on $K$ for Burgers is explored.
- `dataset_visualization.py`: code to illustrate the dataset used by NN-GParareal (nearest neighbors) and the relative distance between observations.
- `NN-GPara_with_time.py`: everything concerning to the "time" extension of NN-GParareal.

Other files:
- `preprocessing.py`: some simulation outputs have been processed to make them lighter, by removing unnecessary information.



## Acknowledgments

Part of the code has been translated and adapted from [GParareal](https://github.com/kpentland/GParareal)'s Matlab implementation .

