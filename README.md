# Neural Network Optimization Using Evolutionary Algorithms

This repository contains the code for a Master’s thesis conducted at the University of Basel, which received the **Best Master’s Thesis Award** for the academic year.

The project presents a from-scratch Python implementation of feed-forward neural networks whose architectures and weights are optimized with evolutionary algorithms. It explores how evolutionary operators and NSGA-II selection can be used to identify models that effectively balance predictive performance and model complexity.

The implementation does **not** use TensorFlow, PyTorch, Keras, or other neural-network frameworks. Core neural-network layers, activation functions, training, mutation, population management, and selection logic are implemented directly with NumPy.

## Project goals

- Build and train simple fully connected neural networks from scratch.
- Generate populations of neural-network architectures.
- Apply evolutionary mutations to network structure and parameters.
- Use NSGA-II multi-objective selection to prefer accurate and compact models.
- Run experiments on:
  - MNIST digit classification data
  - Wholesale customers classification data
  - Synthetic/artificial data

## Repository structure

```text
.
├── EvolutionaryAlgorithm.py      # Population initialization, training, offspring creation, prediction
├── Experiment_ArtificialData.py  # Artificial-data experiment entry point
├── Experiment_MNIST.py           # MNIST experiment entry point
├── Experiment_Wholesale.py       # Wholesale customers experiment entry point
├── NSGAII.py                     # NSGA-II parent selection
├── NeuralNetwork_Batch.py        # NeuralNetwork class and batch training loop
├── Population.py                 # Population container and reporting helpers
└── support/
    ├── AbstractLayer.py          # Base layer abstraction
    ├── ActivationLayer.py        # Activation-layer implementation
    ├── Bootstrap.py              # Data splitting, standardization, label transforms
    ├── Functions.py              # Activation and loss functions
    ├── Layer.py                  # Dense layer implementation
    ├── MutationAction.py         # Evolutionary mutation operators
    ├── Parameters.py             # Random architecture initialization
    ├── evaluation_selection.py   # Non-dominated sorting and objective helpers
    └── plotting_helper.py        # Plotting utilities for experiment results
```

## How the algorithm works

At a high level, each experiment follows this workflow:

1. Load and preprocess a dataset.
2. Create an initial population of randomly initialized neural networks.
3. Train each network with backpropagation and Adam-style weight updates.
4. Mutate copies of the current population to create offspring.
5. Evaluate parent and offspring networks on validation data.
6. Use NSGA-II to select the next parent population.
7. Plot performance, architecture size, test accuracy, and exploration metrics.

The evolutionary mutation operations include:

- Adding a hidden layer
- Removing a hidden layer
- Jittering network weights
- Pruning small weights
- Changing hidden-layer activation functions

The main optimization objectives are validation accuracy and network size, represented by the number of neurons.

## Requirements

The project is a plain Python codebase and does not currently include a `requirements.txt` file. Install the following packages before running experiments:

```bash
pip install numpy pandas scipy matplotlib seaborn
```

Recommended environment:

- Python 3.8 or newer
- A virtual environment such as `venv` or `conda`

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy matplotlib seaborn
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas scipy matplotlib seaborn
```

## Datasets

Datasets are not included in this repository. The experiment scripts currently reference local absolute paths from the original author’s machine, so you must update those paths before running the experiments.

### MNIST

`Experiment_MNIST.py` expects CSV files for the MNIST training and test sets:

```python
dataPath_train = "/path/to/mnist_train.csv"
dataPath_test = "/path/to/mnist_test.csv"
```

Expected format:

- One row per image
- First column: digit label from `0` to `9`
- Remaining columns: pixel values

### Wholesale customers

`Experiment_Wholesale.py` expects the UCI Wholesale customers dataset:

```python
dataPath = "/path/to/Wholesale customers data.csv"
```

Expected format:

- A CSV containing the original Wholesale customers columns
- The script drops the `Region` column and uses the first remaining column as the class label

### Artificial data

`Experiment_ArtificialData.py` expects NumPy arrays:

```python
train_inputs.npy
train_targets.npy
test_inputs.npy
test_targets.npy
```

Update the `np.load(...)` paths in the script to point to your local files.

## Running experiments

After installing dependencies and updating dataset paths, run an experiment from the repository root:

```bash
python Experiment_Wholesale.py
```

```bash
python Experiment_MNIST.py
```

```bash
python Experiment_ArtificialData.py
```

Each script prints generation-by-generation population statistics and opens Matplotlib figures for the collected metrics.

## Configuring experiments

Most experiment settings are currently configured directly in the Python files.

### Population and generations

The experiment scripts control population sizes, number of generations, training epochs, and minimum accuracy thresholds. For example:

- `popSize`
- `it`
- `minAcc`
- `epochs`
- `batchSize`

### Network search bounds

`NeuralNetwork_Batch.py` defines static bounds used when generating and mutating architectures:

- `maxNeurons`
- `minNeurons`
- `maxHiddenLayers`
- `sizeInput`
- `sizeOutput`

The file contains commented presets for MNIST and artificial-data experiments, while the active values are currently configured for the Wholesale customers experiment. Update these values to match the dataset before running another experiment.

## Outputs and plots

The experiment scripts use helper functions from `support/plotting_helper.py` to visualize:

- Objective trade-offs
- Training iterations
- Test accuracy
- Architecture exploration over generations
- Artificial-data experiment results

Plots are displayed interactively with `plt.show()`.

## Troubleshooting

### Dataset path errors

If you see `FileNotFoundError`, update the dataset paths in the experiment script you are running.

### Import errors on case-sensitive systems

Some imports in the experiment files use capitalized module names, while the files in `support/` are lowercase in this repository. If you run on a case-sensitive filesystem and see import errors such as:

```text
ModuleNotFoundError: No module named 'support.Plotting_helper'
ModuleNotFoundError: No module named 'support.Evaluation_Selection'
```

update the imports to match the actual filenames:

- `support.plotting_helper`
- `support.evaluation_selection`

### Matplotlib style errors

The scripts use:

```python
plt.style.use("seaborn-whitegrid")
```

If your Matplotlib version does not recognize that style, install Seaborn or replace the style with one available in your environment.

## Current limitations

- Dataset files are not bundled with the repository.
- Experiment configuration is hard-coded in scripts rather than exposed through command-line arguments.
- There is no automated test suite or packaging configuration.
- The code is intended for experimentation and research rather than production use.

## Contributing

Useful improvements include:

- Adding a `requirements.txt` or environment file
- Replacing hard-coded dataset paths with command-line arguments
- Adding reproducible dataset setup instructions
- Adding automated tests for layers, losses, mutations, and NSGA-II selection
- Normalizing module filenames and imports for cross-platform compatibility
- Adding saved outputs for plots and experiment metrics

## License

No license file is currently included. If you plan to reuse or distribute this project, contact the repository owner for licensing guidance.
