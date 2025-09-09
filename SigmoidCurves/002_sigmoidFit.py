import matplotlib
matplotlib.use("Agg")  # headless backend

import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

# ---------- Configuration ----------
DATA_CSV = Path("Merged_Cleaned_outlier_data.csv")
PLOTS_DIR = Path("plots")
RESULTS_CSV = Path("sigmoidFit.csv")
EARLY_DATES = ["2022/03/10", "2022/03/12", "2022/03/14"]
EARLY_AREA = 0.0081
MANUAL_SKIP_LIST = [
    "2;48;96;6500;426;12 (ID 10)",
    "2;83;166;11231;513;21 (ID 10)",
]
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ---------- Utility Functions ----------
def make_image_list(path: Path) -> List[Path]:
    return sorted(path.glob("*.jpg"))

def read_data(csv_path: Path = DATA_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Line_Rep_id"] = df["name_ID"]
    df["Date"] = pd.to_datetime(df["Date"])  # infer format
    return df

def add_parameters_to_data(df: pd.DataFrame, line_rep_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_filtered = df[df.Line_Rep_id == line_rep_id].copy()
    new_df = pd.DataFrame({
        "Date": pd.to_datetime(EARLY_DATES, format="%Y/%m/%d"),
        "Area": [EARLY_AREA]*len(EARLY_DATES),
        "Line_Rep_id": [line_rep_id]*len(EARLY_DATES),
    })
    df_filtered = pd.concat([df_filtered, new_df], ignore_index=True)
    df_filtered = df_filtered.sort_values(by=["Line_Rep_id", "Date"]).reset_index(drop=True)
    df_filtered["Line_Rep_id"] = df_filtered["Line_Rep_id"].ffill()
    min_date, max_date = df_filtered["Date"].min(), df_filtered["Date"].max()
    all_dates = pd.DataFrame({"Date": pd.date_range(start=min_date, end=max_date)})
    df_merged = all_dates.merge(df_filtered, on="Date", how="left")
    df_unfiltered = df_merged.dropna(subset=["Area"]).copy()
    df_filtered_final = df_unfiltered[df_unfiltered.get("Information") != "Outlier"].copy()
    df_unfiltered["row_number"] = range(1, len(df_unfiltered)+1)
    df_filtered_final["row_number"] = range(1, len(df_filtered_final)+1)
    return df_filtered_final, df_unfiltered

# ---------- Sigmoid & derivatives ----------
def sigmoid(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    return L / (1 + np.exp(-k*(x - x0)))

def sigmoid_first_derivative(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    exp_term = np.exp(-k*(x - x0))
    return (L * k * exp_term) / (1 + exp_term)**2

def sigmoid_second_derivative(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    exp_term = np.exp(-k*(x - x0))
    return (L * k**2 * exp_term * (1 - exp_term)) / (1 + exp_term)**3

# ---------- Fitting ----------
def fit_sigmoid(df_filtered: pd.DataFrame):
    x = df_filtered["row_number"].to_numpy()
    y = df_filtered["Area"].to_numpy()
    p0 = [y.max(), 1.0, np.median(x)]
    x_new = np.arange(x.min(), x.max()+1, 1)
    popt, pcov = curve_fit(sigmoid, x, y, p0=p0, maxfev=500_000)
    y_fit = sigmoid(x_new, *popt)
    return popt, pcov, x_new, y_fit

# ---------- Derivative Analysis ----------
def analyze_derivatives(popt: np.ndarray, x_new: np.ndarray, y_fit: np.ndarray) -> dict:
    L, k, x0 = popt
    first_deriv = sigmoid_first_derivative(x_new, *popt)
    second_deriv = sigmoid_second_derivative(x_new, *popt)
    peaks_idx = argrelextrema(first_deriv, np.greater)[0]
    if peaks_idx.size == 0:
        peaks_idx = np.array([int(np.argmin(np.abs(x_new - x0)))])
    idx_peak = peaks_idx[0]
    x_first_derivative_peak = x_new[idx_peak]
    y_first_derivative_peak = y_fit[idx_peak]
    min_idx = int(np.argmin(second_deriv))
    max_idx = int(np.argmax(second_deriv))
    slope_at_x0 = sigmoid_first_derivative(x0, L, k, x0)
    y_tangent = sigmoid(x0, L, k, x0) + slope_at_x0*(x_new - x0)
    a, b = np.polyfit(x_new, y_tangent, 1)
    return {
        "first_deriv": first_deriv,
        "second_deriv": second_deriv,
        "x_first_derivative_peak": x_first_derivative_peak,
        "y_first_derivative_peak": y_first_derivative_peak,
        "x0": x0,
        "y_tangent": y_tangent,
        "max_second_derivative_x": x_new[max_idx],
        "min_second_derivative_x": x_new[min_idx],
        "a": a,
        "b": b,
    }

# ---------- Plotting ----------
def plot_all(x_new, y_fit, first_deriv, x, y, line_rep_id, x_first_derivative_peak, y_tangent, df_unfiltered):
    fig, ax = plt.subplots()
    ax.plot(x_new, y_fit, label="Fitted curve")
    ax.scatter(x, y, label="Observed points")
    if "Information" in df_unfiltered.columns:
        outliers = df_unfiltered[df_unfiltered.Information == "Outlier"]
        if not outliers.empty:
            ax.scatter(outliers.row_number, outliers.Area, label="Outlier", alpha=0.8)
    ax.axvline(x=x_first_derivative_peak, linestyle="--", label="Peak first derivative")
    plt.savefig(PLOTS_DIR / f"{line_rep_id}.jpg")
    plt.close(fig)

# ---------- CSV Helpers ----------
def reset_results_csv(path: Path = RESULTS_CSV):
    header = ("line,rep,id,line_rep_id,x_for_steepest_slope,y_for_steepest_slope,"
              "a_for_axplusb_steepest_point,b_for_axplusb_steepest_point,"
              "x_at_min_value_sigmoid,y_at_min_value_sigmoid,x_at_max_value_sigmoid,"
              "y_at_max_value_sigmoid,first_day_of_growth,final_day_of_growth\n")
    path.write_text(header)

def write_results_csv(path: Path, values: List[str]):
    path.write_text(path.read_text() + ",".join(values) + "\n")

# ---------- Min/Max ----------
def find_min_max(df: pd.DataFrame, fit: np.ndarray):
    df = df.copy()
    series = pd.Series(fit, index=np.arange(len(fit)))
    df = df.reset_index(drop=True)
    df["fit"] = series
    max_idx = int(series.idxmax())
    min_idx = int(series.idxmin())
    df["diff_to_max"] = (df["row_number"] - max_idx).abs()
    df["diff_to_min"] = (df["row_number"] - min_idx).abs()
    max_row = df.loc[df["diff_to_max"].idxmin()]
    min_row = df.loc[df["diff_to_min"].idxmin()]
    return float(min_row["fit"]), pd.to_datetime(min_row["Date"]), float(max_row["fit"]), pd.to_datetime(max_row["Date"])

# ---------- Main ----------
def main():
    df = read_data()
    image_list = make_image_list(PLOTS_DIR)
    reset_results_csv()
    for name in df.name_ID.unique():
        if any(skip for skip in MANUAL_SKIP_LIST if skip in name):
            continue
        if any(str(name) in p.name for p in image_list):
            continue
        df_filtered, df_unfiltered = add_parameters_to_data(df, name)
        if len(df_filtered) < 5:
            (PLOTS_DIR / f"{name}_didnt_work.jpg").write_text("\n")
            continue
        popt, _, x_new, y_fit = fit_sigmoid(df_filtered)
        min_area, min_date, max_area, max_date = find_min_max(df_filtered, y_fit)
        derivs = analyze_derivatives(popt, x_new, y_fit)
        plot_all(x_new, y_fit, derivs["first_deriv"], df_filtered["row_number"].to_numpy(),
                 df_filtered["Area"].to_numpy(), df_filtered.Line_Rep_id.iloc[-1],
                 derivs["x_first_derivative_peak"], derivs["y_tangent"], df_unfiltered)
        vals = [
            "placeholder","placeholder","placeholder",
            str(df_filtered.Line_Rep_id.iloc[-1]),
            str(derivs["x_first_derivative_peak"]),
            str(derivs["y_first_derivative_peak"]),
            str(derivs["a"]),
            str(derivs["b"]),
            str(min_date),
            str(min_area),
            str(max_date),
            str(max_area),
            str(derivs["min_second_derivative_x"]),
            str(derivs["max_second_derivative_x"]),
        ]
        write_results_csv(RESULTS_CSV, vals)

if __name__ == "__main__":
    main()

