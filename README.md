<h1> Deep Learning–Based Prediction of Polymer Transfection Efficiency </h1>

This repository contains the code used in the study:

“Predicting Polymer Drug Delivery Efficiency Using 3D Molecular Structures and Deep Learning”

which presents a three-dimensional convolutional neural network (3D CNN) framework for predicting the transfection efficiency of polymer-based delivery systems for nucleic acid therapeutics.

**Overview**

The model encodes cationic and hydrophobic polymer segments into 3D voxel representations derived from multiple molecular conformations and predicts transfection efficiency using a 3D CNN architecture.

**Repository Structure**

- voxelize.py # Generate voxel representations from SMILES
- datahandle.py # Utils used for handling dataset
- model.py # 3D CNN-based model architecture
- train.py # Perform training
- requirements.txt # Python dependencies

**Requirements**

The code was developed and tested using:

- Python 3.12
- PyTorch
- NumPy
- RDKit
- molvoxel

**Data Availability**

Due to data sharing restriction, raw experimental data may not be fully publicly available. Processed data used for model training and evaluation can be provided where permitted. Additional data may be available from the corresponding author upon reasonable request.

**Licence**

This repository is released under the MIT License.
