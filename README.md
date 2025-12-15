## A state-of-the-art intrinsic dimension estimater of high dimensional data using quantum cognition machine learning. Leverages JAX to run on GPU/TPU.

**Author**  : Geet Rakala

**License** :  MIT

- By using ideas developed in the fields of quantum cognition and quantum geometry, a new state of the art intrinsic dimension estimator of high dimensional data was proposed : https://www.nature.com/articles/s41598-025-91676-8

- Beyond immediate use as a state-of-the-art intrinsic dimension estimator, this framework is ripe for adopting into quantum computing pipelines as is being attempted by a startup : https://www.qognitiveai.com/

- This repository contains an open source implementation of the this intrinsic dimension estimator. The code is contained in qcml.py and is written in a functional style using JAX. This allows it to run seamlessly on CPU/GPU/TPU or what have you.

- Start by reading the white paper (https://www.nature.com/articles/s41598-025-91676-8)
- All functions and main are defined in qcml.py.
- Various solvers for the diagonalisation can be selected [pseudo, analytic, LBFGS..]
- One of two parametrisation modes can be selected [pauli,upper]
- The generated plots, and comparison to the white paper results can be found in plots.pdf
- The mathematical justification for the pseudo gradient approach can be found in gradient.pdf
