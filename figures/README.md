# Generate figures

Stand-alone notebook to reproduce plots in Figure 1 & 2 on directional startle behavior. For a quick preview, [open the notebook on github](https://github.com/danionella/veith_et_al_2024/blob/main/figures/generate_figures_1_2.ipynb).

## Running the notebook on Colab
You can execute the notebook on a runtime hosted by Google Research (click "Open in Colab") 

<a target="_blank" href="https://colab.research.google.com/github/danionella/veith_et_al_2024/blob/main/figures/generate_figures_1_2.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

or run it on your own Jupyter Notebook environment.


## Running the notebook on a local installation
### Requirements
- There are no specific hadware requirements. This software should run on Windows, Linux and Mac.

### Installation procedure
- Install [conda for Python 3.x](https://github.com/conda-forge/miniforge)
- Change to the directory containing this file
- Execute the collowing lines
```
conda create -n veith2024 python=3.8 numpy scipy pandas matplotlib ipympl -c conda-forge
conda activate veith2024
jupyter notebook generate_figures_1_2.ipynb
```
Above steps typically take 10-15 mins.
