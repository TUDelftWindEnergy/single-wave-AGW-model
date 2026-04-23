# -*- coding: utf-8 -*-
"""
@author: jiaxinliu

User run script for AGW BTS generation.

Edit and run this file:
    python run_agw_cases.py

Modes
-----
1) default
   - generate one safe default case
   - save BTS and plots

2) run_batch
   - generate batch cases
   - optionally plot first N cases
"""

from agw_bts_model import run_from_user_settings


USER_SETTINGS = {
    # =========================================================
    # 1. MODE
    # =========================================================
    "run_mode": "default",   # "default" or "run_batch"

    # =========================================================
    # 2. OUTPUT SETTINGS
    # =========================================================
    "output": {
        "output_base_dir": "wind_field_runs",
        "default_folder_name": "default_debug_case",
        "save_case_matrix_csv": True,
        "save_info_file": True,
        "save_csv_each_timestep": False,
        "copy_run_script_to_output": False
    },

    # =========================================================
    # 3. PLOTTING SETTINGS
    # =========================================================
    "plotting": {
        # Global plot switch
        "enable_plotting": True,

        # In default mode: always plot if enable_plotting=True
        "default_mode_make_plots": True,

        # In batch mode:
        # - if False: do not plot any batch case
        # - if True: plot the first batch_plot_first_n_cases valid cases
        "batch_mode_make_plots": True,
        "batch_plot_first_n_cases": 3,

        # Two time indices used for YZ-plane contour plots
        "snapshot_time_indices": [0, 100],

        # One time index used for wind-profile comparison plot
        "profile_time_index": 0
    },

    # =========================================================
    # 4. GLOBAL SETTINGS
    # =========================================================
    "global_settings": {
        "wind_profile": "PLExp",          # "Uniform", "PLExp", "LOG"
        "branch": "auto",                 # "plus", "minus", "auto"
        "agw_amplitude_mode": "local_U",  # "local_U", "URef"
        "dt": 5.0,
        "total_time": 3660.0
    },

    # =========================================================
    # 5. BATCH PARAMETERS
    # =========================================================
    "batch_parameters": {
        "URef_list": [9.6],
        "Zhigh_list": [88.5],
        "PLExp_list": [0.2],
        "beta_deg_list": [0.0, 30.0, 89.0],
        "omega_list": [0.0, 0.002, 0.006],
        "agw_amp_factor_list": [0.10]
    }
}


if __name__ == "__main__":
    run_from_user_settings(USER_SETTINGS, run_script_path=__file__)