# bci-simulator

This repository contains code used for the paper "Closed-loop motor imagery EEG simulation for brain-computer interfaces". The code was developed under partial support from NIH under grant AT009263.

## Requirements
* pygame (tested on 1.9.6)
* spectrum
* scipy/numpy/pandas
* tkinter

## Usage
1. Run `main.py`, or, for a portable install, use an installation tool (e.g. pyinstaller) on `main.py`.
2. The software is ready to be used for an experiment in default settings. To edit common parameters, change `params = launcher("subj")` to `params = launcher("dev")` to access the GUI upon launch, or edit parameters directly in `ui\get_params.py` or `main.py`.

## Citation
If this software, in whole or in part, helped your project, please consider citing the following paper:

Shin H, Suma D and He B (2022) Closed-loop motor imagery EEG simulation for brain-computer interfaces. *Front. Hum. Neurosci.* 16:951591. doi: 10.3389/fnhum.2022.951591
