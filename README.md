# PSI-ML_2025-protein-folds

*PSI-ML_2025-protein-folds* is a pedagogical resource for the attendees of the Machine Learning for Scientific Research summer school (held in Petnica Science Center in 2025). It is meant for beginners and experts to explore the protein fold space with Graph Neural Networks.

## Installation

Clone the repo and install the conda enviroment. 

```bash
$ git clone https://github.com/VGligorijevic/PSI-ML_2025-protein-folds.git
$ cd PSI-ML_2025-protein-folds
$ conda env create -f env.yml
$ conda activate protfold
```

## Data
To download PDB files of the protein domains from the Protein Structure Classification Databese (CATH), run:

```bash
$ wget ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz
$ tar -xvzf cath-dataset-nonredundant-S40.pdb.tgz 
```

## Contact

Authors: Vladimir Gligorijevic (vgligorijevic@gmail.com)
We welcome your questions and feedback via email or GitHub Issues.
