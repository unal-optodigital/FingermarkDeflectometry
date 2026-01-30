import re
from pathlib import Path
import numpy as np

# ---- Tu función (sin cambios) ----
def michelson_contrast_per_region(minmax_list: list[list[float]]) -> list[float]:
    """
    minmax_list: [[min,max],[min,max],...]
    retorna: [C0, C1, ...] con C = (Imax - Imin)/(Imax + Imin)
    Si Imax+Imin == 0 -> NaN
    """
    out = []
    for mn, mx in minmax_list:
        den = mx + mn
        out.append(float(np.abs((mx - mn)) / den) if den != 0 else float(0))
    return out

# ---- Regex para parsear "region k: min=..., max=..." ----
_REGION_RE = re.compile(
    r"region\s*\d+\s*:\s*min\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+|nan)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"max\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+|nan)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE
)

def parse_minmax_from_txt(txt_path: Path) -> list[list[float]]:
    minmax = []
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _REGION_RE.search(line)
        if not m:
            continue
        mn = float(m.group(1))
        mx = float(m.group(2))
        minmax.append([mn, mx])
    return minmax

def find_region_txt(fingermark_dir: Path, exclude_names: set[str]) -> Path | None:
    """
    Busca el primer .txt que parezca contener regiones min/max.
    Excluye archivos de salida (por nombre).
    """
    txts = sorted([p for p in fingermark_dir.glob("*.txt") if p.name not in exclude_names])
    for p in txts:
        try:
            if len(parse_minmax_from_txt(p)) > 0:
                return p
        except Exception:
            pass
    return None

def process_fingermark_folder(fm_dir: Path) -> dict:
    """
    Calcula contrastes por región, promedio (ignorando NaN),
    y escribe michelson_mean.txt dentro del fingermark.
    """
    out_name = "michelson_mean.txt"
    txt_in = find_region_txt(fm_dir, exclude_names={out_name})

    if txt_in is None:
        (fm_dir / out_name).write_text(
            "No se encontró un .txt con líneas tipo: region k: min=..., max=...\n",
            encoding="utf-8"
        )
        return {
            "fingermark": fm_dir.name,
            "input_txt": None,
            "n_regions": 0,
            "n_valid": 0,
            "mean_contrast": float("nan"),
        }

    minmax = parse_minmax_from_txt(txt_in)
    contrasts = michelson_contrast_per_region(minmax)
    c = np.array(contrasts, dtype=float)

    valid = np.isfinite(c)
    mean_c = float(np.mean(c[valid])) if np.any(valid) else float("nan")

    # Guardar salida por fingermark
    lines = [
        f"input_txt: {txt_in.name}",
        f"n_regions_total: {len(contrasts)}",
        f"n_regions_valid: {int(np.sum(valid))}",
        f"mean_michelson_contrast: {mean_c}",
        "",
        "per_region_contrast:",
    ]
    for i, val in enumerate(contrasts, start=1):
        lines.append(f"region {i}: {val}")

    (fm_dir / out_name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "fingermark": fm_dir.name,
        "input_txt": txt_in.name,
        "n_regions": len(contrasts),
        "n_valid": int(np.sum(valid)),
        "mean_contrast": mean_c,
    }

def main(root_imagenes: str = "imagenes"):
    root = Path(root_imagenes)
    if not root.exists():
        raise FileNotFoundError(f"No existe la carpeta raíz: {root.resolve()}")

    # Resumen global (UNO SOLO)
    global_results = []

    # Experimentos = subcarpetas directas de "imagenes" (ej: 1,2,3,...)
    exp_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    for exp_dir in exp_dirs:
        mic_dir = exp_dir / "michelson_contrast"
        if not mic_dir.exists():
            continue

        fm_dirs = sorted([
            p for p in mic_dir.iterdir()
            if p.is_dir() and p.name.lower().startswith("fingermark")
        ])

        for fm_dir in fm_dirs:
            res = process_fingermark_folder(fm_dir)
            # Añade info de experimento y ruta para el resumen global
            res["experiment"] = exp_dir.name
            res["fingermark_path"] = str(fm_dir)
            global_results.append(res)

    # Escribir resumen global en imagenes/michelson_contrast_summary.txt
    summary_path = root / "michelson_contrast_summary.txt"
    lines = []
    lines.append(f"root: {root.resolve()}")
    lines.append(f"n_total_fingermarks: {len(global_results)}")
    lines.append("")
    lines.append("experiment\tfingermark\tmean_contrast\tn_valid\tn_total\tinput_txt\tpath")

    for r in global_results:
        lines.append(
            f"{r['experiment']}\t{r['fingermark']}\t{r['mean_contrast']}\t"
            f"{r['n_valid']}\t{r['n_regions']}\t{r['input_txt']}\t{r['fingermark_path']}"
        )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Resumen global guardado en: {summary_path}")

if __name__ == "__main__":
    main("imagenes")