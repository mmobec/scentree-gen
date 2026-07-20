![Scentree logo](https://raw.githubusercontent.com/mmobec/scentree/main/docs/assets/card.svg)

# About

`scentree` is a Python framework for generating scenario trees for multistage stochastic programming. It provides methods for generating a scenario fan using statistical models based on multivariate time series and machine learning techniques. The resulting scenario tree is then constructed using scenario reduction algorithms.

If the optimization problem to be solved is related to an energy community participating in electricity markets, the open-source [`secoem`](https://github.com/mmobec/secoem) package can be used after generating the scenario tree with the `scentree` package.


# Installation
The package is currently available only on `PyPI` and can be installed with:
```bash
pip install scentree
```

The package will be available on conda-forge soon.

# How to use it
Visit [the official documentation page](https://scentree.readthedocs.io/en/latest/) for more details on how to use it.

# Citation
```bibtex
@misc{scentree2026,
  author = {Cristian Pachón-García and Albert Solà Vilalta and F.-Javier Heredia},
  title = {{scentree}},
  subtitle = {a framework for generating scenario trees for multistage stochastic programming},
  url = {https://github.com/mmobec/scentree},
  year = {2026}
}
```

# Acknowledgements
This work has been supported with grants PID2022-139219OB-I00 and Cetp-FP-2023-00185 from the Spanish Ministerio de Ciencia, Innovación y Universidades. This research has been funded by CETPartnership, the Clean Energy Transition Partnership under the 2023 joint call for research proposals, co-funded by the European Commission (GA N°101069750) and with the funding organisations FFG (Austria), AEI (Spain) and MUR (Italy).