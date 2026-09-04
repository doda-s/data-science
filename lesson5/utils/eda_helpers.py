"""
Reusable EDA building blocks shared by the lesson5 *_eda.py scripts: plotting
setup, figure saving, descriptive-statistics summaries and IQR-based outlier
counting. Keeping this logic in one place avoids re-writing the same
boilerplate (and re-fixing the same bugs) in every dataset script.

Typical usage in an EDA script:

    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from utils.eda_helpers import setup_output_dir, save_figure, numeric_summary, outlier_report, missing_data_report
    from utils.html_report import build_html_report

    import matplotlib.pyplot as plt
    import pandas
    import seaborn
    import sklearn.datasets as skdatasets

    OUTPUT_DIR = setup_output_dir(__file__, "my_dataset")
    df = skdatasets.fetch_openml(data_id=..., as_frame=True).data

    missing_data_report(df)
    print(numeric_summary(df).round(2))

    seaborn.boxplot(y=df["some_column"])
    save_figure(OUTPUT_DIR, "boxplot_some_column.png")

    build_html_report(OUTPUT_DIR, title="My Dataset EDA")
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas
import seaborn


def setup_output_dir(script_file, subdir):
    """
    Creates (if needed) and returns lesson5/output/<subdir>/, the folder where
    a script's plots and HTML report are written. Also applies the shared
    seaborn theme used across every lesson5 script.
    """
    output_dir = os.path.join(os.path.dirname(script_file), "output", subdir)
    os.makedirs(output_dir, exist_ok=True)
    seaborn.set_theme(style="whitegrid")
    return output_dir


def save_figure(output_dir, filename):
    """Applies tight_layout, saves the current matplotlib figure and closes it."""
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    print(f"Grafico salvo em {path}")


def count_outliers_iqr(series):
    """Number of values in `series` outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return ((series < lower_bound) | (series > upper_bound)).sum()


def outlier_report(df, columns=None):
    """
    DataFrame with the IQR-based outlier count and percentage of rows it
    represents, for each of `columns` (defaults to every numeric column),
    sorted from the most- to the least-affected column.
    """
    if columns is None:
        columns = df.select_dtypes("number").columns

    outliers_per_column = pandas.Series(
        {c: count_outliers_iqr(df[c]) for c in columns}
    ).sort_values(ascending=False)
    outlier_pct = (outliers_per_column / len(df) * 100).round(2)
    return pandas.DataFrame({"n_outliers": outliers_per_column, "pct_do_dataframe": outlier_pct})


def numeric_summary(df, columns=None):
    """
    Central-tendency and dispersion table (mean, mode, median, std, variance,
    quartiles, skewness) for `columns` (defaults to every numeric column).

    Note: `mode` is only meaningful when values actually repeat. For columns
    where every value is unique (nunique == len(df)), `.mode()` just returns
    the smallest value tied at frequency 1 -- check `df[col].nunique()` before
    reading the mode as if it were informative.
    """
    if columns is None:
        columns = df.select_dtypes("number").columns

    summary = df[columns].describe().T
    summary["mode"] = [df[c].mode().iloc[0] for c in columns]
    summary["variance"] = df[columns].var()
    summary["skewness"] = df[columns].skew()
    summary = summary[["mean", "mode", "50%", "std", "variance", "min", "25%", "75%", "max", "skewness"]]
    summary.columns = ["mean", "mode", "median", "std", "variance", "min", "Q1", "Q3", "max", "skewness"]
    return summary


def missing_data_report(df):
    """Prints null counts per column, the total, and the duplicated-row count."""
    null_counts = df.isna().sum()
    print(null_counts)
    print(f"Total de valores ausentes: {null_counts.sum()}")
    print(f"Linhas duplicadas: {df.duplicated().sum()}")
    return null_counts
