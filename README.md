# Safe Mutations for Deep and Recurrent Neural Networks through Output Gradients 

Code for a pytorch implementation of the mutation operators from [the paper](https://arxiv.org/abs/1712.06563). 

This repository contains code for the illustrative experiments, the recurrent parity task, and the breadcrumb hard maze task. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

* [PyTorch](http://pytorch.org/) -- version 0.2.0_2 was what was used in the experiment
* [Python](http://python.org/) -- version 2.7
* [PyGame](https://www.pygame.org/) -- only relevant for visualization
* [SCons](http://scons.org/) -- to build the hard maze domain

### Building

If you want to run the breadcrumb hard maze experiment, you will need to install [SCons](http://scons.org) and run the command ``scons`` in the working directory, which should build from source the python module. Note that the current release requires python 2.7 (although it is likely possible to port it to python 3). 

### Running the Illustrative Experiments

```sm_simple.py``` contains code to recreate the illustrative experiments, and can be invoked in the
following way:

```python sm_simple.py --domain [domain] --mutation [method] --mutation_mag [float]```

The parameter **domain** can be one of: easy, medium, or washout (for the gradient-washout task). 

The parameter **mutation** can be one of: control, SM-G-SUM, SM-G-SO, SM-G-ABS, or SM-R. 

The parameter **mutation_mag** can be set to an arbitrary float value. Parameter values used in the paper's experiments can be found in the appendix.

### Running Other Experiments

```experiment_runner.py``` contains code to run experiments in the recurrent parity task and the breadcrumb hard maze. It can be invoked in the following way:

``python experiment_runner.py --domain [domain] --mutation [method] --mutation_mag [float]```

The parameter **domain** can be one of: classification (for the recurrent parity task) or breadcrumb_maze (for the breadcrumb_hard_maze).

The parameter **mutation** can be one of: control, SM-G-SUM, SM-G-SO, SM-G-ABS, or SM-R. 

The parameter **mutation_mag** can be set to an arbitrary float value. Parameter values used in the paper's experiments can be found in the appendix.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

