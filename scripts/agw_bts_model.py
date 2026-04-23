# -*- coding: utf-8 -*-
"""
@author: jiaxinliu

Backend model for AGW wind-field and BTS generation.
"""

import os
import itertools
import shutil
import numpy as np
import pandas as pd
from turbsim_file import TurbSimFile
from agw_plot_utils import make_all_case_plots


INTERNAL_DEFAULTS = {
    "physics": {
        "N": 0.010
    },
    "grid": {
        "y_min": -200.0,
        "y_max": 200.0,
        "z_min": 0.0,
        "z_max": 400.0,
        "delta_n": 5.0
    },
    "log_profile": {
        "Z0": 0.03,
        "Zmin_log": 1.0
    },
    "recommendation": {
        "target_wavelength_min": 50.0,
        "target_wavelength_max": 2000.0
    },
    "default_case": {
        "URef": 9,
        "Zhigh": 88.5,
        "PLExp": 0.2,
        "beta_deg": 45.0,
        "omega": 0.002,
        "agw_amp_factor": 0.10
    }
}


def deep_update(default_dict, user_dict):
    merged = default_dict.copy()
    for key, value in user_dict.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def fmt_num(val, decimals=3, strip=True):
    s = f"{val:.{decimals}f}"
    if strip:
        s = s.rstrip("0").rstrip(".")
    return s


def compact_profile_tag(wind_profile):
    if wind_profile == "Uniform":
        return "uni"
    elif wind_profile == "PLExp":
        return "pl"
    elif wind_profile == "LOG":
        return "log"
    raise ValueError(f"Unknown wind_profile: {wind_profile}")


def compact_amplitude_tag(amode):
    if amode == "local_U":
        return "lu"
    elif amode == "URef":
        return "ur"
    raise ValueError(f"Unknown amplitude mode: {amode}")


def make_case_name(wind_profile, URef, Zhigh, PLExp, beta_deg, omega,
                   agw_amp_factor, branch, agw_amplitude_mode):
    bg = compact_profile_tag(wind_profile)
    am = compact_amplitude_tag(agw_amplitude_mode)
    return (
        f"agw_{bg}"
        f"_U{fmt_num(URef, 2)}"
        f"_Zh{fmt_num(Zhigh, 2)}"
        f"_a{fmt_num(PLExp, 3)}"
        f"_b{fmt_num(beta_deg, 1)}"
        f"_om{fmt_num(omega, 4)}"
        f"_amp{fmt_num(agw_amp_factor, 3)}"
        f"_{branch}_{am}"
    )


def validate_positive_number(value, name):
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number. Got: {value}")


def validate_nonempty_list(value, name):
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty list.")


def validate_user_settings(user_settings):
    run_mode = user_settings.get("run_mode", None)
    if run_mode not in ["default", "run_batch"]:
        raise ValueError('run_mode must be "default" or "run_batch".')

    global_settings = user_settings.get("global_settings", {})
    plotting = user_settings.get("plotting", {})
    batch_parameters = user_settings.get("batch_parameters", {})

    if "wind_profile" in global_settings:
        if global_settings["wind_profile"] not in ["Uniform", "PLExp", "LOG"]:
            raise ValueError('global_settings.wind_profile must be "Uniform", "PLExp", or "LOG".')

    if "branch" in global_settings:
        if global_settings["branch"] not in ["plus", "minus", "auto"]:
            raise ValueError('global_settings.branch must be "plus", "minus", or "auto".')

    if "agw_amplitude_mode" in global_settings:
        if global_settings["agw_amplitude_mode"] not in ["local_U", "URef"]:
            raise ValueError('global_settings.agw_amplitude_mode must be "local_U" or "URef".')

    if "dt" in global_settings:
        validate_positive_number(global_settings["dt"], "global_settings.dt")
    if "total_time" in global_settings:
        validate_positive_number(global_settings["total_time"], "global_settings.total_time")

    if "batch_plot_first_n_cases" in plotting:
        if plotting["batch_plot_first_n_cases"] < 0:
            raise ValueError("plotting.batch_plot_first_n_cases must be >= 0.")

    if "snapshot_time_indices" in plotting:
        if len(plotting["snapshot_time_indices"]) != 2:
            raise ValueError("plotting.snapshot_time_indices must contain exactly 2 integers.")

    if run_mode == "run_batch":
        for key in [
            "URef_list", "Zhigh_list", "PLExp_list",
            "beta_deg_list", "omega_list", "agw_amp_factor_list"
        ]:
            if key not in batch_parameters:
                raise ValueError(f'Missing batch parameter list: "{key}"')
            validate_nonempty_list(batch_parameters[key], f"batch_parameters.{key}")


def uniform_profile(z, uref):
    return np.full_like(z, uref, dtype=float)


def power_law_profile(z, uref, zhigh, plexp):
    z_safe = np.maximum(z, 1e-6)
    return uref * (z_safe / zhigh) ** plexp


def logarithmic_profile(z, uref, zhigh, z0, zmin=1.0):
    z_use = np.maximum(z, zmin)
    return uref * np.log(z_use / z0) / np.log(zhigh / z0)


def compute_background_wind(Z, wind_profile, URef, Zhigh, PLExp, Z0, Zmin_log):
    if wind_profile == "Uniform":
        return uniform_profile(Z, URef)
    elif wind_profile == "PLExp":
        return power_law_profile(Z, URef, Zhigh, PLExp)
    elif wind_profile == "LOG":
        return logarithmic_profile(Z, URef, Zhigh, Z0, Zmin_log)
    raise ValueError(f"Unknown wind_profile: {wind_profile}")


def compute_amplitude_field(U_bg, URef, agw_amp_factor, agw_amplitude_mode):
    if agw_amplitude_mode == "local_U":
        return agw_amp_factor * U_bg
    elif agw_amplitude_mode == "URef":
        return np.full_like(U_bg, agw_amp_factor * URef, dtype=float)
    raise ValueError("agw_amplitude_mode must be 'local_U' or 'URef'")


def compute_k_and_lambda(branch_name, omega, N, URef, cos_beta):
    if abs(cos_beta) < 1e-12:
        raise ValueError("cos(beta) is too close to zero.")

    if branch_name == "plus":
        k_val = (omega + N) / (URef * cos_beta)
    elif branch_name == "minus":
        k_val = (omega - N) / (URef * cos_beta)
    else:
        raise ValueError("branch_name must be 'plus' or 'minus'.")

    lam_val = np.inf if abs(k_val) < 1e-12 else 2.0 * np.pi / abs(k_val)
    return k_val, lam_val


def branch_score(k_val, lam_val, target_min, target_max):
    if not np.isfinite(lam_val) or abs(k_val) < 1e-12:
        return -999
    score = 0
    score += 2 if k_val > 0 else -1
    score += 3 if target_min <= lam_val <= target_max else -1
    return score


def recommend_branch(omega, N, URef, beta_deg, target_min, target_max):
    beta = np.radians(beta_deg)
    cos_beta = np.cos(beta)

    k_plus, lam_plus = compute_k_and_lambda("plus", omega, N, URef, cos_beta)
    k_minus, lam_minus = compute_k_and_lambda("minus", omega, N, URef, cos_beta)

    s_plus = branch_score(k_plus, lam_plus, target_min, target_max)
    s_minus = branch_score(k_minus, lam_minus, target_min, target_max)

    if s_plus > s_minus:
        rec = "plus"
    elif s_minus > s_plus:
        rec = "minus"
    else:
        rec = "undetermined"

    return {
        "k_plus": k_plus,
        "lambda_plus": lam_plus,
        "k_minus": k_minus,
        "lambda_minus": lam_minus,
        "recommended_branch": rec
    }


def select_branch(branch_setting, recommended_branch):
    if branch_setting in ["plus", "minus"]:
        return branch_setting
    if branch_setting == "auto":
        return recommended_branch if recommended_branch in ["plus", "minus"] else "plus"
    raise ValueError("branch_setting must be plus, minus, or auto.")


def build_grid(grid_cfg):
    y_min = grid_cfg["y_min"]
    y_max = grid_cfg["y_max"]
    z_min = grid_cfg["z_min"]
    z_max = grid_cfg["z_max"]
    delta_n = grid_cfg["delta_n"]

    y_vals = np.arange(y_min, y_max + 0.5 * delta_n, delta_n)
    z_vals = np.arange(z_min, z_max + 0.5 * delta_n, delta_n)
    Y, Z = np.meshgrid(y_vals, z_vals, indexing="ij")
    return y_vals, z_vals, Y, Z


def save_grid_csv(output_dir, Y, Z):
    grid_df = pd.DataFrame({
        "y_coordinate (m)": Y.flatten(order="C"),
        "z_coordinate (m)": Z.flatten(order="C")
    })
    grid_df.to_csv(os.path.join(output_dir, "grid_generated.csv"), index=False)


def generate_single_case(
    case_id,
    total_cases,
    case_dir,
    case_name,
    params,
    merged_cfg,
    Y,
    Z,
    y_vals,
    z_vals,
    turb_time,
    save_csv_each_timestep,
    save_info_file,
    make_plots_for_this_case=False
):
    wind_profile = merged_cfg["global_settings"]["wind_profile"]
    branch_setting = merged_cfg["global_settings"]["branch"]
    agw_amplitude_mode = merged_cfg["global_settings"]["agw_amplitude_mode"]
    dt = merged_cfg["global_settings"]["dt"]
    total_time = merged_cfg["global_settings"]["total_time"]

    plotting = merged_cfg["plotting"]

    N = merged_cfg["physics"]["N"]
    Z0 = merged_cfg["log_profile"]["Z0"]
    Zmin_log = merged_cfg["log_profile"]["Zmin_log"]
    target_wavelength_min = merged_cfg["recommendation"]["target_wavelength_min"]
    target_wavelength_max = merged_cfg["recommendation"]["target_wavelength_max"]

    URef = params["URef"]
    Zhigh = params["Zhigh"]
    PLExp = params["PLExp"]
    beta_deg = params["beta_deg"]
    omega = params["omega"]
    agw_amp_factor = params["agw_amp_factor"]

    beta = np.radians(beta_deg)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)

    if abs(cos_beta) < 1e-8:
        print(f"[Skip] Case {case_id}: beta={beta_deg} makes cos(beta) singular.")
        return None

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

    A_min = float(np.min(A_field))
    A_max = float(np.max(A_field))

    rec_info = recommend_branch(
        omega=omega,
        N=N,
        URef=URef,
        beta_deg=beta_deg,
        target_min=target_wavelength_min,
        target_max=target_wavelength_max
    )

    selected_branch = select_branch(branch_setting, rec_info["recommended_branch"])

    if selected_branch == "plus":
        k = rec_info["k_plus"]
        wavelength = rec_info["lambda_plus"]
    else:
        k = rec_info["k_minus"]
        wavelength = rec_info["lambda_minus"]

    if not np.isfinite(wavelength) or abs(k) < 1e-12:
        print(f"[Skip] Case {case_id}: selected k is singular or nearly zero.")
        return None

    num_timesteps = len(turb_time)
    Ny = len(y_vals)
    Nz = len(z_vals)

    u_data = np.zeros((num_timesteps, Ny, Nz), dtype=np.float32)
    v_data = np.zeros((num_timesteps, Ny, Nz), dtype=np.float32)
    w_data = np.zeros((num_timesteps, Ny, Nz), dtype=np.float32)

    print(
        f"[{case_id:03d}/{total_cases:03d}] {case_name} | "
        f"branch={selected_branch} | k={k:.6f}, lambda={wavelength:.2f} m"
    )

    for t_idx in range(num_timesteps):
        t = t_idx * dt

        phase = k * (Z * sin_beta) - omega * t
        cos_phase = np.cos(phase)

        u_prime = A_field * sin_beta * cos_phase
        v_prime = np.zeros_like(U_bg)
        w_prime = -A_field * cos_beta * cos_phase

        U_total = U_bg + u_prime
        V_total = v_prime
        W_total = w_prime

        u_data[t_idx, :, :] = U_total.astype(np.float32)
        v_data[t_idx, :, :] = V_total.astype(np.float32)
        w_data[t_idx, :, :] = W_total.astype(np.float32)

        if save_csv_each_timestep:
            data_out = np.column_stack((
                Y.flatten(order="C"),
                Z.flatten(order="C"),
                U_total.flatten(order="C"),
                V_total.flatten(order="C"),
                W_total.flatten(order="C")
            ))
            csv_filename = os.path.join(case_dir, f"wind_t{t_idx:03d}.csv")
            header = (
                f"y (m), z (m), U (m/s), V (m/s), W (m/s) "
                f"[AGW+BG on YZ plane at x=0, t={t:.1f}s]"
            )
            np.savetxt(csv_filename, data_out, delimiter=",", comments="", header=header, fmt="%.6f")

    output_bts_file = os.path.join(case_dir, f"{case_name}.bts")

    dst = TurbSimFile()
    dst["z"] = z_vals.astype(np.float32)
    dst["y"] = y_vals.astype(np.float32)
    dst["t"] = turb_time
    dst["u"] = np.stack([u_data, v_data, w_data], axis=0)
    dst.write(output_bts_file)

    if save_info_file:
        info_file = os.path.join(case_dir, "INFO.txt")
        with open(info_file, "w", encoding="utf-8") as f:
            f.write("=== AGW BTS case ===\n\n")
            f.write(f"case_id = {case_id}\n")
            f.write(f"case_name = {case_name}\n\n")

            f.write("[Parameters]\n")
            f.write(f"wind_profile = {wind_profile}\n")
            f.write(f"URef = {URef}\n")
            f.write(f"Zhigh = {Zhigh}\n")
            f.write(f"PLExp = {PLExp}\n")
            f.write(f"beta_deg = {beta_deg}\n")
            f.write(f"omega = {omega}\n")
            f.write(f"agw_amp_factor = {agw_amp_factor}\n")
            f.write(f"N = {N}\n")
            f.write(f"branch_setting = {branch_setting}\n")
            f.write(f"selected_branch = {selected_branch}\n")
            f.write(f"agw_amplitude_mode = {agw_amplitude_mode}\n")
            f.write(f"k = {k}\n")
            f.write(f"wavelength = {wavelength}\n")
            f.write(f"A_min = {A_min}\n")
            f.write(f"A_max = {A_max}\n\n")

    if make_plots_for_this_case and plotting["enable_plotting"]:
        make_all_case_plots(
            case_dir=case_dir,
            t=turb_time,
            y_vals=y_vals,
            z_vals=z_vals,
            U_bg=U_bg,
            u_data=u_data,
            w_data=w_data,
            profile_time_index=plotting["profile_time_index"],
            snapshot_time_indices=plotting["snapshot_time_indices"],
            hub_height=88.5
        )

    return {
        "case_id": case_id,
        "case_name": case_name,
        "wind_profile": wind_profile,
        "URef": URef,
        "Zhigh": Zhigh,
        "PLExp": PLExp,
        "beta_deg": beta_deg,
        "omega": omega,
        "agw_amp_factor": agw_amp_factor,
        "N": N,
        "branch_setting": branch_setting,
        "selected_branch": selected_branch,
        "agw_amplitude_mode": agw_amplitude_mode,
        "k": k,
        "wavelength": wavelength,
        "k_plus": rec_info["k_plus"],
        "lambda_plus": rec_info["lambda_plus"],
        "k_minus": rec_info["k_minus"],
        "lambda_minus": rec_info["lambda_minus"],
        "recommended_branch": rec_info["recommended_branch"],
        "A_min": A_min,
        "A_max": A_max,
        "dt": dt,
        "total_time": total_time,
        "num_timesteps": num_timesteps,
        "bts_file": output_bts_file
    }


def run_from_user_settings(user_settings, run_script_path=None):
    validate_user_settings(user_settings)
    merged_cfg = deep_update(INTERNAL_DEFAULTS, user_settings)

    run_mode = merged_cfg["run_mode"]
    output_cfg = merged_cfg["output"]
    plotting = merged_cfg["plotting"]
    global_settings = merged_cfg["global_settings"]

    output_base_dir = output_cfg["output_base_dir"]
    default_folder_name = output_cfg["default_folder_name"]
    save_case_matrix_csv = output_cfg["save_case_matrix_csv"]
    save_info_file = output_cfg["save_info_file"]
    save_csv_each_timestep = output_cfg["save_csv_each_timestep"]
    copy_run_script_to_output = output_cfg["copy_run_script_to_output"]

    dt = global_settings["dt"]
    total_time = global_settings["total_time"]
    num_timesteps = int(total_time / dt) + 1
    turb_time = np.arange(0.0, num_timesteps * dt, dt, dtype=np.float32)

    os.makedirs(output_base_dir, exist_ok=True)

    y_vals, z_vals, Y, Z = build_grid(merged_cfg["grid"])
    save_grid_csv(output_base_dir, Y, Z)

    if copy_run_script_to_output and run_script_path is not None and os.path.isfile(run_script_path):
        shutil.copy2(run_script_path, os.path.join(output_base_dir, os.path.basename(run_script_path)))

    case_records = []

    print("=== AGW BTS generation ===")
    print(f"run_mode = {run_mode}")
    print(f"output_base_dir = {output_base_dir}")

    if run_mode == "default":
        p = merged_cfg["default_case"]

        case_name = make_case_name(
            wind_profile=global_settings["wind_profile"],
            URef=p["URef"],
            Zhigh=p["Zhigh"],
            PLExp=p["PLExp"],
            beta_deg=p["beta_deg"],
            omega=p["omega"],
            agw_amp_factor=p["agw_amp_factor"],
            branch=global_settings["branch"],
            agw_amplitude_mode=global_settings["agw_amplitude_mode"]
        )

        case_dir = os.path.join(output_base_dir, default_folder_name)
        os.makedirs(case_dir, exist_ok=True)

        record = generate_single_case(
            case_id=1,
            total_cases=1,
            case_dir=case_dir,
            case_name=case_name,
            params=p,
            merged_cfg=merged_cfg,
            Y=Y,
            Z=Z,
            y_vals=y_vals,
            z_vals=z_vals,
            turb_time=turb_time,
            save_csv_each_timestep=save_csv_each_timestep,
            save_info_file=save_info_file,
            make_plots_for_this_case=plotting["default_mode_make_plots"]
        )
        if record is not None:
            case_records.append(record)

    elif run_mode == "run_batch":
        batch = merged_cfg["batch_parameters"]

        param_combinations = list(itertools.product(
            batch["URef_list"],
            batch["Zhigh_list"],
            batch["PLExp_list"],
            batch["beta_deg_list"],
            batch["omega_list"],
            batch["agw_amp_factor_list"]
        ))

        total_cases = len(param_combinations)
        print(f"Total number of requested cases = {total_cases}")

        valid_case_counter = 0

        for case_id, combo in enumerate(param_combinations, start=1):
            URef, Zhigh, PLExp, beta_deg, omega, agw_amp_factor = combo

            params = {
                "URef": URef,
                "Zhigh": Zhigh,
                "PLExp": PLExp,
                "beta_deg": beta_deg,
                "omega": omega,
                "agw_amp_factor": agw_amp_factor
            }

            case_name = make_case_name(
                wind_profile=global_settings["wind_profile"],
                URef=URef,
                Zhigh=Zhigh,
                PLExp=PLExp,
                beta_deg=beta_deg,
                omega=omega,
                agw_amp_factor=agw_amp_factor,
                branch=global_settings["branch"],
                agw_amplitude_mode=global_settings["agw_amplitude_mode"]
            )

            case_dir = os.path.join(output_base_dir, case_name)
            os.makedirs(case_dir, exist_ok=True)

            make_plots = False
            if plotting["batch_mode_make_plots"]:
                if valid_case_counter < plotting["batch_plot_first_n_cases"]:
                    make_plots = True

            record = generate_single_case(
                case_id=case_id,
                total_cases=total_cases,
                case_dir=case_dir,
                case_name=case_name,
                params=params,
                merged_cfg=merged_cfg,
                Y=Y,
                Z=Z,
                y_vals=y_vals,
                z_vals=z_vals,
                turb_time=turb_time,
                save_csv_each_timestep=save_csv_each_timestep,
                save_info_file=save_info_file,
                make_plots_for_this_case=make_plots
            )

            if record is not None:
                case_records.append(record)
                valid_case_counter += 1

    if save_case_matrix_csv:
        case_matrix_df = pd.DataFrame(case_records)
        case_matrix_path = os.path.join(output_base_dir, "case_matrix.csv")
        case_matrix_df.to_csv(case_matrix_path, index=False)
        print(f"[Saved] Case matrix: {case_matrix_path}")

    print("\n✅ Completed.")
    print(f"Generated {len(case_records)} valid case(s).")