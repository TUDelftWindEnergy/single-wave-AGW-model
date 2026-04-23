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
   └─ turbsim_file.py"# single-wave-AGW-model" 
