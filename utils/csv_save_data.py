import re
from pathlib import Path
import numpy as np
import pandas as pd


def michelson_contrast_per_region(minmax_list):
    """
    minmax_list: [[min,max],[min,max],...]
    returns: [C0, C1, ...] con C = |Imax - Imin| / (Imax + Imin)
    if Imax+Imin == 0 -> 0
    """
    out = []
    for mn, mx in minmax_list:
        den = mx + mn
        out.append(float(np.abs(mx - mn) / den) if den != 0 else 0.0)
    return out


def parse_minmax_txt(txt_path):
    """
    read files:
    region 1: min=126.97, max=139.21
    ...
    y returns:
    [[min1,max1], [min2,max2], ...]
    """
    pattern = re.compile(
        r"region\s+\d+\s*:\s*min\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*max\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
    )

    minmax_list = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                mn = float(m.group(1))
                mx = float(m.group(2))
                minmax_list.append([mn, mx])

    if not minmax_list:
        raise ValueError(f"Could extract regions: {txt_path}")

    return minmax_list


def summarize_contrast_file(txt_path):
    """
    file contrast_*.txt:
    - parse min/max each region
    - calculates contrast Michelson per region
    - retorns dict 
    """
    minmax_list = parse_minmax_txt(txt_path)
    contrasts = np.array(michelson_contrast_per_region(minmax_list), dtype=float)

    n = len(contrasts)
    mean = float(np.mean(contrasts)) if n > 0 else np.nan
    std = float(np.std(contrasts, ddof=1)) if n > 1 else 0.0
    sem = float(std / np.sqrt(n)) if n > 0 else np.nan

    return {
        "n_regions": n,
        "mean_contrast": mean,
        "std_contrast": std,
        "sem_contrast": sem,
        "region_contrasts": contrasts.tolist(),
    }


def collect_fingermark_summaries(general_folder):
    """
    folders type fingermark_1, fingermark_2, ..., fingermark_N
    generate tables:
    - amplitude
    - phase
    """
    general_folder = Path(general_folder)

    if not general_folder.exists():
        raise FileNotFoundError(f"No Folder: {general_folder}")

    rows_amp = []
    rows_phase = []

    fingermark_dirs = sorted(
        [p for p in general_folder.iterdir() if p.is_dir()],
        key=lambda p: int(re.search(r"(\d+)$", p.name).group(1)) if re.search(r"(\d+)$", p.name) else 10**9
    )

    for folder in fingermark_dirs:
        m = re.search(r"(\d+)$", folder.name)
        if not m:
            print(f": {folder.name}")
            continue

        idx = int(m.group(1))

        amp_path = folder / f"contrast_amplitude{idx}.txt"
        phase_path = folder / f"contrast_phase{idx}.txt"

        if amp_path.exists():
            s = summarize_contrast_file(amp_path)
            rows_amp.append({
                "X_Pos": idx,
                "Y_Value": s["mean_contrast"],
                "Std": s["std_contrast"],
                "SEM": s["sem_contrast"],
                "N": s["n_regions"],
                **{f"Region_{i+1}": v for i, v in enumerate(s["region_contrasts"])}
            })
        else:
            print(f"No existe: {amp_path}")

        if phase_path.exists():
            s = summarize_contrast_file(phase_path)
            rows_phase.append({
                "X_Pos": idx,
                "Y_Value": s["mean_contrast"],
                "Std": s["std_contrast"],
                "SEM": s["sem_contrast"],
                "N": s["n_regions"],
                **{f"Region_{i+1}": v for i, v in enumerate(s["region_contrasts"])}
            })
        else:
            print(f"No exists: {phase_path}")

    df_amp = pd.DataFrame(rows_amp).sort_values("X_Pos").reset_index(drop=True)
    df_phase = pd.DataFrame(rows_phase).sort_values("X_Pos").reset_index(drop=True)

    return df_amp, df_phase


def save_outputs(general_folder, df_amp, df_phase):
    general_folder = Path(general_folder)

    amp_csv = general_folder / "amplitude_summary.csv"
    phase_csv = general_folder / "phase_summary.csv"

    df_amp.to_csv(amp_csv, index=False, encoding="utf-8")
    df_phase.to_csv(phase_csv, index=False, encoding="utf-8")

    print(f"Guardado: {amp_csv}")
    print(f"Guardado: {phase_csv}")
    return amp_csv, phase_csv


