# Single-Wave AGW Model

This repository contains a simplified engineering workflow for generating atmospheric gravity wave (AGW) wind fields on a YZ inflow plane and exporting them as BTS files for wind turbine simulations.

The model is designed for controlled parameter studies of AGW effects on inflow structure, turbine response, and related aerodynamic or aeroelastic analyses. It supports both a quick default test case and batch generation over user-defined AGW parameter combinations.

---

## Model overview

This model represents a simplified single-wave AGW superimposed on a background wind profile.

Main assumptions:
- The inflow plane is the **YZ plane at x = 0**
- The AGW is represented as a simplified wave in the **XZ plane**
- The background wind can be:
  - Uniform
  - Power-law profile (`PLExp`)
  - Logarithmic profile (`LOG`)
- The AGW perturbation is added to the background wind field
- The generated wind field is exported as a **BTS file** for later use in simulation tools such as OpenFAST

---

## Main execution scripts

There are two user-facing execution scripts in this repository.

### Wind-data generation script

#### `scripts/run_agw_cases.py`

This is the main script used to generate AGW wind data.

Run this script when you want to:
- generate BTS wind-field files
- run a default test case
- run a batch of AGW parameter combinations
- optionally generate plots during the case-generation process

In short, this is the script for **wind-data generation**.

### Plotting script

#### `scripts/plot_existing_cases.py`

This is the script used for plotting existing generated cases.

Run this script when you want to:
- re-plot selected cases after generation
- make figures for specific parameter combinations
- regenerate plots without rerunning the full BTS generation workflow

In short, this is the script for **post-processing and plotting**.

---

## Source of `turbsim_file.py`

The file `scripts/turbsim_file.py` is not developed originally in this repository. It is adapted from the `turbsim_file.py` implementation available in the OpenFAST Toolbox / weio-related file I/O utilities.

Source:
- OpenFAST Toolbox repository
- Path: `openfast_toolbox/io/turbsim_file.py`

If this repository is used in publications, reports, or shared research workflows, please acknowledge the original source of `turbsim_file.py` accordingly.

---

## Repository structure

A typical repository structure is:

```text
single-wave-AGW-model/
│
├─ README.md
└─ scripts/
   ├─ agw_bts_model.py
   ├─ agw_plot_utils.py
   ├─ plot_existing_cases.py
   ├─ run_agw_cases.py
   └─ turbsim_file.py
```

---

## Author

Jiaxin Liu  
Delft University of Technology
