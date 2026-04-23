# -*- coding: utf-8 -*-
"""
@author: jiaxinliu

Plot existing AGW cases by selecting cases from case_matrix.csv.

Run:
    python plot_existing_cases.py
"""

import os
import pandas as pd
import numpy as np
from agw_bts_model import (
    INTERNAL_DEFAULTS,
    deep_update,
    build_grid,
    compute_background_wind,
    compute_amplitude_field,
    recommend_branch,
    select_branch
)
from agw_plot_utils import make_all_case_plots


PLOT_SETTINGS = {
    "output_base_dir": "wind_field_runs",
    "case_matrix_filename": "case_matrix.csv",

    # Filter cases by exact values; use None to ignore a field
    "filters": {
        "URef": 9.6,
        "Zhigh": 88.5,
        "PLExp": 0.2,
        "beta_deg": 30.0,
        "omega": 0.002,
        "agw_amp_factor": 0.10
    },

    "global_settings": {
        "wind_profile": "PLExp",
        "branch": "auto",
        "agw_amplitude_mode": "local_U",
        "dt": 5.0,
        "total_time": 3660.0
    },

    "plotting": {
        "profile_time_index": 0,
        "snapshot_time_indices": [0, 100]
    }
}


def case_matches_filters(row, filters):
    for key, value in filters.items():
        if value is None:
            continue
        if row[key] != value:
            return False
    return True


def regenerate_case_fields(row, cfg):
    y_vals, z_vals, Y, Z = build_grid(cfg["grid"])

    dt = cfg["global_settings"]["dt"]
    total_time = cfg["global_settings"]["total_time"]
    num_timesteps = int(total_time / dt) + 1
    t = np.arange(0.0, num_timesteps * dt, dt, dtype=np.float32)

    URef = row["URef"]
    Zhigh = row["Zhigh"]
    PLExp = row["PLExp"]
    beta_deg = row["beta_deg"]
    omega = row["omega"]
    agw_amp_factor = row["agw_amp_factor"]

    wind_profile = cfg["global_settings"]["wind_profile"]
    agw_amplitude_mode = cfg["global_settings"]["agw_amplitude_mode"]
    branch_setting = cfg["global_settings"]["branch"]

    N = cfg["physics"]["N"]
    Z0 = cfg["log_profile"]["Z0"]
    Zmin_log = cfg["log_profile"]["Zmin_log"]
    target_wavelength_min = cfg["recommendation"]["target_wavelength_min"]
    target_wavelength_max = cfg["recommendation"]["target_wavelength_max"]

    beta = np.radians(beta_deg)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)

    U_bg = compute_background_wind(
        Z=Z,
        wind_profile=wind_profile,
        URef=URef,
        Zhigh=Zhigh,
        PLExp=PLExp,
        Z0=Z0,
        Zmin_log=Zmin_log
    )

    A_field = compute_amplitude_field(
        U_bg=U_bg,
        URef=URef,
        agw_amp_factor=agw_amp_factor,
        agw_amplitude_mode=agw_amplitude_mode
    )

    rec = recommend_branch(
        omega=omega,
        N=N,
        URef=URef,
        beta_deg=beta_deg,
        target_min=target_wavelength_min,
        target_max=target_wavelength_max
    )

    selected_branch = select_branch(branch_setting, rec["recommended_branch"])
    k = rec["k_plus"] if selected_branch == "plus" else rec["k_minus"]

    Ny = len(y_vals)
    Nz = len(z_vals)
    Nt = len(t)

    u_data = np.zeros((Nt, Ny, Nz), dtype=np.float32)
    w_data = np.zeros((Nt, Ny, Nz), dtype=np.float32)

    for tidx in range(Nt):
        tt = tidx * dt
        phase = k * (Z * sin_beta) - omega * tt
        cos_phase = np.cos(phase)

        u_prime = A_field * sin_beta * cos_phase
        w_prime = -A_field * cos_beta * cos_phase

        u_data[tidx, :, :] = (U_bg + u_prime).astype(np.float32)
        w_data[tidx, :, :] = w_prime.astype(np.float32)

    return t, y_vals, z_vals, U_bg, u_data, w_data


def main():
    output_base_dir = PLOT_SETTINGS["output_base_dir"]
    case_matrix_path = os.path.join(output_base_dir, PLOT_SETTINGS["case_matrix_filename"])

    if not os.path.isfile(case_matrix_path):
        raise FileNotFoundError(f"Cannot find case matrix: {case_matrix_path}")

    df = pd.read_csv(case_matrix_path)
    filters = PLOT_SETTINGS["filters"]

    cfg = deep_update(INTERNAL_DEFAULTS, PLOT_SETTINGS)

    matched = df[df.apply(lambda row: case_matches_filters(row, filters), axis=1)]

    if len(matched) == 0:
        print("No cases matched the specified filters.")
        return

    print(f"Matched {len(matched)} case(s).")

    for _, row in matched.iterrows():
        case_name = row["case_name"]
        case_dir = os.path.join(output_base_dir, case_name)

        if not os.path.isdir(case_dir):
            print(f"[Skip] Missing case folder: {case_dir}")
            continue

        print(f"Plotting case: {case_name}")

        t, y_vals, z_vals, U_bg, u_data, w_data = regenerate_case_fields(row, cfg)

        make_all_case_plots(
            case_dir=case_dir,
            t=t,
            y_vals=y_vals,
            z_vals=z_vals,
            U_bg=U_bg,
            u_data=u_data,
            w_data=w_data,
            profile_time_index=cfg["plotting"]["profile_time_index"],
            snapshot_time_indices=cfg["plotting"]["snapshot_time_indices"],
            hub_height=88.5
        )


if __name__ == "__main__":
    main()