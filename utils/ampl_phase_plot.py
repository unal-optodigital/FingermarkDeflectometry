import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
from pathlib import Path
# REQUIREMENTS
# DATA PATH: surface_results/Surface/images/michelson_contrast/CSV

# REQUIRED CVS: amplitude_summary.cvs     grading.cvs      phase_summary.csv



def make_multi_amplitude_plot(
    amplitude_csv_paths,
    labels=None,
    show_errorbars=True,
    errorbar_type="SEM",       # "SEM", "Std", None
    curve_mode="fit",          # "connect", "fit", "none"
    fit_type="poly",           # "poly" o "exp"
    fit_degree=3,
    figsize=(12, 7),
    xlabel="Depletion n°",
    ylabel="Michelson contrast",
    title="",
    x_tick_step=1,
    marker_size=6,
    line_width=2.0,
    save_path_png=None,
    save_path_pdf=None
):
    # =========================================================
    # 1. HELPERS
    # =========================================================
    def fit_curve(x, y, fit_type="poly", degree=2, n_points=400):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

        x_smooth = np.linspace(x.min(), x.max(), n_points)

        if fit_type == "poly":
            coeffs = np.polyfit(x, y, degree)
            poly = np.poly1d(coeffs)
            y_smooth = poly(x_smooth)
            return x_smooth, y_smooth

        elif fit_type == "exp":
            mask = y > 0
            x_fit = x[mask]
            y_fit = y[mask]

            if len(x_fit) < 2:
                raise ValueError("not enought points for exp adjust.")

            coeffs = np.polyfit(x_fit, np.log(y_fit), 1)
            b = coeffs[0]
            ln_a = coeffs[1]
            a = np.exp(ln_a)

            y_smooth = a * np.exp(b * x_smooth)
            return x_smooth, y_smooth

        else:
            raise ValueError("fit_type debe ser 'poly' o 'exp'.")

    def get_error_values(df, show_errorbars=True, errorbar_type="SEM"):
        if not show_errorbars or errorbar_type is None:
            return None

        if errorbar_type not in df.columns:
            raise ValueError(
                f"The column '{errorbar_type}' doesnt exist the CSV. "
                f"able columns: {list(df.columns)}"
            )
        return df[errorbar_type].to_numpy(dtype=float)

    def plot_curve_by_mode(ax, x, y, xs, ys, mode, color, linestyle, linewidth, label):
        if mode == "connect":
            idx = np.argsort(x)
            ax.plot(
                np.asarray(x)[idx], np.asarray(y)[idx],
                color=color, linestyle=linestyle, linewidth=linewidth, label=label
            )
        elif mode == "fit":
            ax.plot(
                xs, ys,
                color=color, linestyle=linestyle, linewidth=linewidth, label=label
            )
        elif mode == "none":
            # solo marcadores, la leyenda se maneja aparte
            pass
        else:
            raise ValueError("mode must be 'connect', 'fit' o 'none'.")

    # =========================================================
    # 2. READ ALL CSV
    # =========================================================
    dataframes = []
    for p in amplitude_csv_paths:
        df = pd.read_csv(p)

        required_cols = {"X_Pos", "Y_Value"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"columns left in {p}: {missing}")

        dataframes.append(df)

    if labels is None:
        labels = [Path(p).parent.name for p in amplitude_csv_paths]

    if len(labels) != len(dataframes):
        raise ValueError("labels has to have the same lenght than amplitude_csv_paths")

    # =========================================================
    # 3. STYLE
    # =========================================================
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-."]
    marker = "o"

    title_fontsize = 16
    axis_label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 11
    axis_spine_width = 1.3
    marker_edge_width = 1.2

    errorbar_capsize = 4
    errorbar_linewidth = 1.2
    errorbar_capthick = 1.2

    # =========================================================
    # 4. X RANGE
    # =========================================================
    x_min = min(df["X_Pos"].min() for df in dataframes)
    x_max = max(df["X_Pos"].max() for df in dataframes)

    # =========================================================
    # 5. FIGURE
    # =========================================================
    fig, ax = plt.subplots(figsize=figsize)

    # =========================================================
    # 6. PLOT ALL DATASETS
    # =========================================================
    for i, (df, label) in enumerate(zip(dataframes, labels)):
        x = df["X_Pos"].to_numpy()
        y = df["Y_Value"].to_numpy()
        yerr = get_error_values(df, show_errorbars=show_errorbars, errorbar_type=errorbar_type)

        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]

        # curve
        xs, ys = fit_curve(x, y, fit_type=fit_type, degree=fit_degree)
        plot_curve_by_mode(ax, x, y, xs, ys, curve_mode, color, linestyle, line_width, label)

        # markers
        ax.plot(
            x, y,
            linestyle="None",
            marker=marker,
            markersize=marker_size,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=marker_edge_width,
            color=color
        )

        # error bars
        if show_errorbars and yerr is not None:
            ax.errorbar(
                x, y,
                yerr=yerr,
                fmt="none",
                ecolor=color,
                elinewidth=errorbar_linewidth,
                capsize=errorbar_capsize,
                capthick=errorbar_capthick,
                zorder=1
            )

    # =========================================================
    # 7. FORMAT
    # =========================================================
    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)

    if title:
        ax.set_title(title, fontsize=title_fontsize)

    ax.tick_params(axis="both", labelsize=tick_fontsize, direction="in")

    for spine in ax.spines.values():
        spine.set_linewidth(axis_spine_width)

    xticks = np.arange(x_min, x_max + x_tick_step, x_tick_step)
    ax.set_xticks(xticks)
    ax.set_xlim(0.5 * x_min, 1.02 * x_max)

    ax.legend(fontsize=legend_fontsize)
    plt.tight_layout()

    if save_path_png is not None:
        plt.savefig(save_path_png, dpi=1000, bbox_inches="tight")
    if save_path_pdf is not None:
        plt.savefig(save_path_pdf, bbox_inches="tight")

    plt.show()


def make_csv_plot(csv_path_1, csv_path_2, grading_csv_path=None):
    # =========================================================
    # 1. HELPERS
    # =========================================================
    def normalize_grade(g):
        g = str(g).strip()
        g = g.replace("–", "-").replace("—", "-")

        if g in ["=", "+-", "+/-", "±"]:
            return "+-"
        elif g == "+":
            return "+"
        elif g == "-":
            return "-"
        else:
            return g

    def grade_display(g):
        if g == "+-":
            return "±"
        return g

    def fit_curve(x, y, fit_type="poly", degree=2, n_points=400):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

        x_smooth = np.linspace(x.min(), x.max(), n_points)

        if fit_type == "poly":
            coeffs = np.polyfit(x, y, degree)
            poly = np.poly1d(coeffs)
            y_smooth = poly(x_smooth)
            label_fit = f"Poly deg {degree}"
            return x_smooth, y_smooth, label_fit

        elif fit_type == "exp":
            mask = y > 0
            x_fit = x[mask]
            y_fit = y[mask]

            if len(x_fit) < 2:
                raise ValueError("No hay suficientes puntos positivos para ajuste exponencial.")

            coeffs = np.polyfit(x_fit, np.log(y_fit), 1)
            b = coeffs[0]
            ln_a = coeffs[1]
            a = np.exp(ln_a)

            y_smooth = a * np.exp(b * x_smooth)
            label_fit = r"Exp fit: $ae^{bx}$"
            return x_smooth, y_smooth, label_fit

        else:
            raise ValueError("fit_type debe ser 'poly' o 'exp'.")

    def get_error_values(df, show_errorbars=True, errorbar_type="SEM"):
        if not show_errorbars or errorbar_type is None:
            return None

        if errorbar_type not in df.columns:
            raise ValueError(
                f"La columna '{errorbar_type}' no existe en el CSV. "
                f"Columnas disponibles: {list(df.columns)}"
            )

        return df[errorbar_type].to_numpy(dtype=float)

    # =========================================================
    # 2. READ CSV
    # =========================================================
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    required_cols = {"X_Pos", "Y_Value"}
    missing1 = required_cols - set(df1.columns)
    missing2 = required_cols - set(df2.columns)

    if missing1:
        raise ValueError(f"Faltan columnas en csv_path_1: {missing1}")
    if missing2:
        raise ValueError(f"Faltan columnas en csv_path_2: {missing2}")

    # =========================================================
    # 3. READ GRADING CSV
    # =========================================================
    use_band = grading_csv_path is not None

    if use_band:
        df_grade = pd.read_csv(grading_csv_path)

        required_grade_cols = {"X_Pos", "Grade_amplitude", "Grade_phase"}
        missingg = required_grade_cols - set(df_grade.columns)
        if missingg:
            raise ValueError(
                f"Faltan columnas en grading_csv_path: {missingg}"
            )

        df_grade["Grade_amplitude"] = df_grade["Grade_amplitude"].apply(normalize_grade)
        df_grade["Grade_phase"] = df_grade["Grade_phase"].apply(normalize_grade)

        # unir grading con summaries por X_Pos
        df1 = pd.merge(df1, df_grade[["X_Pos", "Grade_amplitude"]], on="X_Pos", how="left")
        df2 = pd.merge(df2, df_grade[["X_Pos", "Grade_phase"]], on="X_Pos", how="left")

        df1 = df1.rename(columns={"Grade_amplitude": "Grade"})
        df2 = df2.rename(columns={"Grade_phase": "Grade"})

        bad1 = ~df1["Grade"].isin(["+", "+-", "-"])
        bad2 = ~df2["Grade"].isin(["+", "+-", "-"])

        if bad1.any():
            print("Valores de Grade no válidos en amplitud:")
            print(df1.loc[bad1, ["X_Pos", "Grade"]])

        if bad2.any():
            print("Valores de Grade no válidos en fase:")
            print(df2.loc[bad2, ["X_Pos", "Grade"]])

    x1 = df1["X_Pos"].to_numpy()
    y1 = df1["Y_Value"].to_numpy()

    x2 = df2["X_Pos"].to_numpy()
    y2 = df2["Y_Value"].to_numpy()

    # =========================================================
    # 4. CURVE MODE CONTROLS
    # =========================================================
    curve_mode_1 = "fit"       # "connect", "fit", "none"
    curve_mode_2 = "fit"

    fit_type_1 = "poly"
    fit_degree_1 = 4

    fit_type_2 = "poly"
    fit_degree_2 = 4

    # =========================================================
    # 5. ERROR BAR CONTROLS
    # =========================================================
    show_errorbars = True
    errorbar_type = "SEM"      # "SEM", "Std", None

    errorbar_capsize = 4
    errorbar_linewidth = 1.2
    errorbar_capthick = 1.2

    # =========================================================
    # 6. STYLE CONTROLS
    # =========================================================
    title_fontsize = 16
    axis_label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 11
    band_ytick_fontsize = 10
    band_text_size = 11

    line_width_1 = 2.5
    line_width_2 = 2.5
    axis_spine_width = 1.3
    main_marker_edge_width = 1.2
    band_box_edge_width = 1.0

    band_box_size = 220
    band_axis_ratio = 1.8
    band_row_height = 1.0

    marker_size = 6

    figsize = (12, 7)
    main_axis_ratio = 5

    xlabel = "Depletion n°"
    ylabel = "Michelson contrast"
    title = ""

    show_grid = False

    x_tick_step = 1
    x_min = min(df1["X_Pos"].min(), df2["X_Pos"].min())
    x_max = max(df1["X_Pos"].max(), df2["X_Pos"].max())

    # =========================================================
    # 7. APPEARANCE SETTINGS
    # =========================================================
    label1 = "b(x,y)"
    color1 = "blue"
    line_style1 = "solid"
    marker_face_1 = "white"
    marker_edge_1 = "blue"

    label2 = "Δϕ(x,y)"
    color2 = "black"
    line_style2 = "--"
    marker_face_2 = "white"
    marker_edge_2 = "black"

    grade_marker_map = {
        "+":  "o",
        "+-": "o",
        "-":  "o"
    }

    grade_color_map = {
        "+":  "#4daf4a",
        "+-": "#ffb000",
        "-":  "#ed7e7e"
    }

    # =========================================================
    # 8. FITS
    # =========================================================
    xs1, ys1, fit_label_1 = fit_curve(x1, y1, fit_type=fit_type_1, degree=fit_degree_1)
    xs2, ys2, fit_label_2 = fit_curve(x2, y2, fit_type=fit_type_2, degree=fit_degree_2)

    yerr1 = get_error_values(df1, show_errorbars=show_errorbars, errorbar_type=errorbar_type)
    yerr2 = get_error_values(df2, show_errorbars=show_errorbars, errorbar_type=errorbar_type)

    # =========================================================
    # 9. CREATE FIGURE
    # =========================================================
    if use_band:
        fig, (ax, ax_band) = plt.subplots(
            2, 1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={
                "height_ratios": [main_axis_ratio, band_axis_ratio],
                "hspace": 0.05
            }
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_band = None

    # =========================================================
    # 10. CURVES
    # =========================================================
    def plot_curve_by_mode(ax, x, y, xs, ys, mode, color, linestyle, linewidth, label):
        if mode == "connect":
            idx = np.argsort(x)
            ax.plot(np.asarray(x)[idx], np.asarray(y)[idx],
                    color=color, linestyle=linestyle, linewidth=linewidth, label=label)
        elif mode == "fit":
            ax.plot(xs, ys,
                    color=color, linestyle=linestyle, linewidth=linewidth, label=label)
        elif mode == "none":
            pass
        else:
            raise ValueError("mode debe ser 'connect', 'fit' o 'none'.")

    plot_curve_by_mode(ax, x1, y1, xs1, ys1, curve_mode_1, color1, line_style1, line_width_1, label1)
    plot_curve_by_mode(ax, x2, y2, xs2, ys2, curve_mode_2, color2, line_style2, line_width_2, label2)

    # =========================================================
    # 11. MAIN MARKERS
    # =========================================================
    if use_band:
        for grade, marker in grade_marker_map.items():
            sub = df1[df1["Grade"] == grade]
            ax.plot(
                sub["X_Pos"], sub["Y_Value"],
                linestyle="None",
                marker=marker,
                markersize=marker_size,
                markerfacecolor=marker_face_1,
                markeredgecolor=marker_edge_1,
                markeredgewidth=main_marker_edge_width,
                color=color1
            )

        for grade, marker in grade_marker_map.items():
            sub = df2[df2["Grade"] == grade]
            marker = 's'
            ax.plot(
                sub["X_Pos"], sub["Y_Value"],
                linestyle="None",
                marker=marker,
                markersize=marker_size,
                markerfacecolor=marker_face_2,
                markeredgecolor=marker_edge_2,
                markeredgewidth=main_marker_edge_width,
                color=color2
            )
    else:
        ax.plot(x1, y1, linestyle="None", marker="o", markersize=marker_size,
                markerfacecolor=marker_face_1, markeredgecolor=marker_edge_1,
                markeredgewidth=main_marker_edge_width, color=color1)
        ax.plot(x2, y2, linestyle="None", marker="o", markersize=marker_size,
                markerfacecolor=marker_face_2, markeredgecolor=marker_edge_2,
                markeredgewidth=main_marker_edge_width, color=color2)

    # =========================================================
    # 12. ERROR BARS
    # =========================================================
    if show_errorbars and yerr1 is not None:
        ax.errorbar(
            x1, y1, yerr=yerr1,
            fmt="none", ecolor=color1,
            elinewidth=errorbar_linewidth,
            capsize=errorbar_capsize,
            capthick=errorbar_capthick,
            zorder=1
        )

    if show_errorbars and yerr2 is not None:
        ax.errorbar(
            x2, y2, yerr=yerr2,
            fmt="none", ecolor=color2,
            elinewidth=errorbar_linewidth,
            capsize=errorbar_capsize,
            capthick=errorbar_capthick,
            zorder=1
        )

    # =========================================================
    # 13. FORMAT
    # =========================================================
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize)

    ax.tick_params(axis="both", labelsize=tick_fontsize, direction="in")

    if show_grid:
        ax.grid(True, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(axis_spine_width)

    dataset_handles = [
        Line2D([0], [0], color=color1, lw=line_width_1,
               marker='o', markerfacecolor='white', markeredgecolor=marker_edge_1,
               markeredgewidth=main_marker_edge_width, label=label1),
        Line2D([0], [0], color=color2, lw=line_width_2,
               marker='s', markerfacecolor='white', markeredgecolor=marker_edge_2,
               markeredgewidth=main_marker_edge_width, label=label2)
    ]

    leg1 = ax.legend(handles=dataset_handles, fontsize=legend_fontsize, loc="upper right")
    ax.add_artist(leg1)

    if use_band:
        # grade_handles = [
        #     Line2D([0], [0], color='black', label='UNIL Grading', lw=2),
        #     Line2D([0], [0], color="#4daf4a", lw=1,
        #            marker=grade_marker_map["+"], markersize=7,
        #            markerfacecolor='white', markeredgecolor="#4daf4a",
        #            label='(+)'),
        #     Line2D([0], [0], color="#ffb000", lw=1,
        #            marker=grade_marker_map["+-"], markersize=7,
        #            markerfacecolor='white', markeredgecolor="#ffb000",
        #            label='(±)'),
        #     Line2D([0], [0], color="#ed7e7e", lw=1,
        #            marker=grade_marker_map["-"], markersize=7,
        #            markerfacecolor='white', markeredgecolor="#ed7e7e",
        #            label='(-)')
        # ]
        # ax.legend(handles=grade_handles, fontsize=legend_fontsize, loc="upper center", ncol=4)
        print(' ')
    # =========================================================
    # 14. LOWER GRADE AXIS
    # =========================================================
    if use_band:
        y_band_1 = 0.6 * band_row_height
        y_band_2 = 0 * band_row_height

        for _, row in df1.iterrows():
            grade = row["Grade"]
            ax_band.scatter(
                row["X_Pos"], y_band_1,
                s=band_box_size, marker="s",
                facecolor=grade_color_map[grade],
                edgecolor="black", linewidth=band_box_edge_width
            )
            ax_band.text(
                row["X_Pos"], y_band_1,
                grade_display(grade),
                ha="center", va="center",
                fontsize=band_text_size, color="black"
            )

        for _, row in df2.iterrows():
            grade = row["Grade"]
            ax_band.scatter(
                row["X_Pos"], y_band_2,
                s=band_box_size, marker="s",
                facecolor=grade_color_map[grade],
                edgecolor="black", linewidth=band_box_edge_width
            )
            ax_band.text(
                row["X_Pos"], y_band_2,
                grade_display(grade),
                ha="center", va="center",
                fontsize=band_text_size, color="black"
            )

        ax_band.set_yticks([y_band_1, y_band_2])
        ax_band.set_yticklabels([label1, label2], fontsize=band_ytick_fontsize)
        ax_band.set_xlabel(xlabel, fontsize=axis_label_fontsize)
        ax_band.tick_params(axis="x", labelsize=tick_fontsize, direction="in")
        ax_band.tick_params(axis="y", length=0)
        ax_band.set_ylim(-0.8 * band_row_height, 1.8 * band_row_height)

        for spine in ax_band.spines.values():
            spine.set_linewidth(axis_spine_width)

        ax_band.spines["top"].set_visible(False)
        ax_band.spines["right"].set_visible(False)

        xticks = np.arange(x_min, x_max + x_tick_step, x_tick_step)
        ax_band.set_xticks(xticks)
        ax.set_xlim(0.5 * x_min, 1.02 * x_max)
    else:
        ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
        xticks = np.arange(x_min, x_max + x_tick_step, x_tick_step)
        ax.set_xticks(xticks)
        ax.set_xlim(0.5 * x_min, 1.02 * x_max)

    plt.tight_layout()
    plt.savefig(michelson_folder + "/contrast_with_errorbars_and_grade.png", dpi=1000, bbox_inches="tight")
    plt.savefig(michelson_folder + "/contrast_with_errorbars_and_grade.pdf", bbox_inches="tight")
    plt.show()





global_script_path = os.path.dirname(__file__)
base_folder = Path(global_script_path) / "surface_results" / "Donors_csv"

amplitude_csv_paths = [
    str(base_folder / str(i) / "amplitude_summary.csv")
    for i in range(1, 7)
]


labels = [f"Donor {i}" for i in range(1, 7)]


"""
This if only for different donors. Uncomment if need to apply 
"""
# make_multi_amplitude_plot(             
#     amplitude_csv_paths=amplitude_csv_paths,
#     labels=labels,
#     show_errorbars=True,
#     errorbar_type="SEM",      # o "Std"
#     curve_mode="fit",         # "connect", "fit", "none"
#     fit_type="poly",          # "poly" o "exp"
#     fit_degree=4,
#     figsize=(12, 7),
#     xlabel="Depletion n°",
#     ylabel="Michelson contrast",
#     title="Amplitude comparison across donors",
#     x_tick_step=1,
#     save_path_png=str(base_folder / "all_amplitudes.png"),
#     save_path_pdf=str(base_folder / "all_amplitudes.pdf")
# )


surface_type = "" # Aluminium_surface, Coffecup, Sink, Stainless steel table top
print(surface_type)
michelson_folder = global_script_path +"/surface_results/" + surface_type + f"/images/michelson_contrast"
csv_path_1 = michelson_folder +"/amplitude_summary.csv"
csv_path_2 = michelson_folder +"/phase_summary.csv"
grading_csv_path = michelson_folder +"/grading.csv"
# from csv_save_data import collect_fingermark_summaries, save_outputs
# df_amp, df_phase = collect_fingermark_summaries(michelson_folder)
# amp_csv, phase_csv = save_outputs(michelson_folder, df_amp, df_phase)
# print("\nResumen amplitud:");print(df_amp.head())
# print("\nResumen fase:");print(df_phase.head())

# make_csv_plot(csv_path_1, csv_path_2, grading_csv_path)