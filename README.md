# PINNs
Solving non-linear PDE. Will be developed to solve another problems.

Physics-Informed Neural Networks (PINNs) for solving non-linear partial differential equations (PDEs).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Layout](#project-layout)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository provides a modular library for building and training PINNs. It offers
core neural network components, domain geometry primitives, training utilities and
visualization helpers. The library can be used to solve a variety of PDE-based
problems.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<user>/PINNs.git
cd PINNs
pip install -r requirements.txt
```

For development installation:

```bash
pip install -e .
```

## Project Layout

```
src/                 Core library modules
├─ core/             Neural network and problem definitions
├─ geometry/         Domain and geometry utilities
├─ training/         Training loop and strategies
├─ utils/            Plotting and helper functions
example/             Example Python scripts
notebooks/           Demonstration notebooks
```

## Usage

The library can be imported as `src` or installed and imported as `pinn_project`.
Below is a minimal example showing how a problem and trainer can be configured:

```python
import src as pinns

def residual(coords, u_pred):
    # define PDE residual
    ...

domain = pinns.RectangularDomain(x_range=(0, 1), y_range=(0, 1))
problem = pinns.PDEProblem(residual_fn=residual, boundary_conditions={"dirichlet": 0.0})
model = pinns.NeuralNet(input_dim=2, hidden_dim=80, output_dim=1, num_hidden_layers=4)
trainer = pinns.Trainer(model=model, problem=problem, domain=domain)
trainer.train(epochs=2000)
```

## Examples

- `example/p_laplacian.py` – solves the 2D p-Laplacian Poisson equation.
- `notebooks/p_laplace.ipynb` – interactive notebook version of the p-Laplacian example.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for new
features, bug fixes or documentation improvements.