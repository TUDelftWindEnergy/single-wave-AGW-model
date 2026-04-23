# -*- coding: utf-8 -*-
"""
@author: jiaxinliu

Plot utilities for AGW BTS cases.

Plots included
--------------
1) Hub-height time series
2) Wind profile comparison at one selected time:
   - AGW perturbation
   - Background wind
   - Full wind
3) Two YZ-plane contour plots of full wind speed magnitude
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def find_nearest_index(array, value):
    array = np.asarray(array)
    return int(np.argmin(np.abs(array - value)))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_hub_height_time_series(case_dir, t, y_vals, z_vals, u_data, w_data, hub_height=88.5):
    """
    Plot hub-height time series at centerline y = 0.
    """
    iy = find_nearest_index(y_vals, 0.0)
    iz = find_nearest_index(z_vals, hub_height)

    U_hub = u_data[:, iy, iz]
    W_hub = w_data[:, iy, iz]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, U_hub, label="U at hub height")
    ax.plot(t, W_hub, label="W at hub height")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(f"Hub-height time series at y=0 m, z≈{z_vals[iz]:.2f} m")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(case_dir, "plot_hub_height_time_series.png"), dpi=200)
    plt.close(fig)


def plot_wind_profiles(case_dir, z_vals, U_bg, u_data, profile_time_index, y_vals):
    """
    Plot AGW perturbation, background wind, and full wind profiles together.
    Use centerline y = 0.
    """
    iy = find_nearest_index(y_vals, 0.0)
    tidx = max(0, min(profile_time_index, u_data.shape[0] - 1))

    bg_profile = U_bg[iy, :]
    full_profile = u_data[tidx, iy, :]
    agw_profile = full_profile - bg_profile

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(agw_profile, z_vals, label="AGW perturbation (u')")
    ax.plot(bg_profile, z_vals, label="Background wind")
    ax.plot(full_profile, z_vals, label="Full wind")
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"Wind profiles at y=0 m, time index={tidx}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(case_dir, "plot_wind_profiles.png"), dpi=200)
    plt.close(fig)


def plot_yz_speed_contours(case_dir, y_vals, z_vals, u_data, w_data, time_indices):
    """
    Plot two YZ-plane contour plots of full wind speed magnitude:
        speed = sqrt(U^2 + W^2)
    """
    if len(time_indices) != 2:
        raise ValueError("time_indices must contain exactly two integers.")

    for i, tidx in enumerate(time_indices, start=1):
        tidx = max(0, min(tidx, u_data.shape[0] - 1))
        speed = np.sqrt(u_data[tidx, :, :]**2 + w_data[tidx, :, :]**2)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.pcolormesh(y_vals, z_vals, speed.T, shading="auto")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Wind speed magnitude (m/s)")
        ax.set_xlabel("y (m)")
        ax.set_ylabel("z (m)")
        ax.set_title(f"YZ-plane wind-speed contour, time index={tidx}")
        fig.tight_layout()
        fig.savefig(os.path.join(case_dir, f"plot_yz_speed_contour_t{i}_idx{tidx}.png"), dpi=200)
        plt.close(fig)


def make_all_case_plots(
    case_dir,
    t,
    y_vals,
    z_vals,
    U_bg,
    u_data,
    w_data,
    profile_time_index=0,
    snapshot_time_indices=(0, 1),
    hub_height=88.5
):
    """
    Make all three requested plots for one case.
    """
    ensure_dir(case_dir)

    plot_hub_height_time_series(
        case_dir=case_dir,
        t=t,
        y_vals=y_vals,
        z_vals=z_vals,
        u_data=u_data,
        w_data=w_data,
        hub_height=hub_height
    )

    plot_wind_profiles(
        case_dir=case_dir,
        z_vals=z_vals,
        U_bg=U_bg,
        u_data=u_data,
        profile_time_index=profile_time_index,
        y_vals=y_vals
    )

    plot_yz_speed_contours(
        case_dir=case_dir,
        y_vals=y_vals,
        z_vals=z_vals,
        u_data=u_data,
        w_data=w_data,
        time_indices=snapshot_time_indices
    )

