"""
VEP Statistical Analysis

Outputs:
1. Age vs N75-P100 amplitude graphs for 12x16 and 48x64.
2. Age vs P100 latency graphs for 12x16 and 48x64.
3. Pearson r, p-value and R^2 for each age graph.
4. Descriptive statistics by age group.
5. Correlation table for height, weight and head size.
6. Sex comparison table using Welch's independent-samples t-test.

The left and right eye values are averaged for each participant.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# =========================
# SETTINGS: This section defines the input dataset and establishes the location where all generated statistical results
# and figures will be stored.
# =========================

INPUT_FILE = Path("VEP Results (Final).xlsx").resolve()
OUTPUT_DIR = INPUT_FILE.parent / "VEP_Simplified_Analysis"
FIGURE_DIR = OUTPUT_DIR / "Figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# COLUMN NAMES: This section defines the variable names corresponding to the demographic, anthropometric, P100 latency,
# and N75-P100 amplitude data contained within the input dataset.
# =========================

AGE = "Age"
SEX = "Sex"
HEIGHT = "Height/cm"
WEIGHT = "Weight/kg"
HEAD_SIZE = "Head Size/cm"

LT_12_LAT = "LT_12x16_P100_latency_ms"
RT_12_LAT = "RT_12x16_P100_latency_ms"
LT_12_AMP = "LT_12x16_N75_P100_amplitude_uV"
RT_12_AMP = "RT_12x16_N75_P100_amplitude_uV"

LT_48_LAT = "LT_48x64_P100_latency_ms"
RT_48_LAT = "RT_48x64_P100_latency_ms"
LT_48_AMP = "LT_48x64_N75_P100_amplitude_uV"
RT_48_AMP = "RT_48x64_N75_P100_amplitude_uV"

# =========================
# DATA PREPARATION: This section loads the VEP dataset, converts the relevant variables to numerical format, calculates
# the mean left-right P100 latency and N75-P100 amplitude for each check size, and assigns each participant to the
# appropriate age group.
# =========================

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. "
            "Place the Excel file in the same folder as this script."
        )

    df = pd.read_excel(INPUT_FILE)

    numeric_columns = [
        AGE, HEIGHT, WEIGHT, HEAD_SIZE,
        LT_12_LAT, RT_12_LAT, LT_12_AMP, RT_12_AMP,
        LT_48_LAT, RT_48_LAT, LT_48_AMP, RT_48_AMP,
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["P100_Latency_12x16_ms"] = df[[LT_12_LAT, RT_12_LAT]].mean(axis=1)
    df["N75_P100_Amplitude_12x16_uV"] = df[[LT_12_AMP, RT_12_AMP]].mean(axis=1)

    df["P100_Latency_48x64_ms"] = df[[LT_48_LAT, RT_48_LAT]].mean(axis=1)
    df["N75_P100_Amplitude_48x64_uV"] = df[[LT_48_AMP, RT_48_AMP]].mean(axis=1)

    def assign_age_group(age):
        if 16 <= age <= 25:
            return "16-25"
        elif 26 <= age <= 40:
            return "26-40"
        elif 41 <= age <= 60:
            return "41-60"
        elif age >= 61:
            return "61+"
        return np.nan

    df["Age_Group"] = df[AGE].apply(assign_age_group)

    return df

# =========================
# AGE CORRELATION GRAPHS: This section generates the scatter plots with linear regression lines to assess the
# relationship between age and each VEP measurement, while calculating the corresponding Pearson correlation coefficient
# (r), p-value, and coefficient of determination (R^2).
# =========================

def create_age_graph(df, outcome, ylabel, filename):
    sub = df[[AGE, outcome]].dropna()

    x = sub[AGE].to_numpy()
    y = sub[outcome].to_numpy()

    r_value, p_value = stats.pearsonr(x, y)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    r_squared = r_value ** 2

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = intercept + slope * x_line

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        x,
        y,
        c="black",
        marker="o",
        s=35
    )

    ax.plot(
        x_line,
        y_line,
        color="black",
        linestyle="-",
        linewidth=1.5
    )

    ax.set_xlabel("Age (years)")
    ax.set_ylabel(ylabel)

    stats_text = (
        f"$r$ = {r_value:.3f}\n"
        f"$p$ = {p_value:.3f}\n"
        f"$R^2$ = {r_squared:.3f}"
    )

    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left"
    )

    ax.grid(False)
    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / filename,
        dpi=500,
        bbox_inches="tight"
    )

    plt.close(fig)

    return {
        "Outcome": outcome,
        "N": len(sub),
        "Pearson_r": r_value,
        "p_value": p_value,
        "R_squared": r_squared,
        "Slope_per_year": slope,
        "Statistically_Significant": "Yes" if p_value < 0.05 else "No"
    }

# =========================
# DESCRIPTIVE STATISTICS BY AGE GROUP: This section calculates the mean and standard deviation (SD) of the P100 latency
# and N75-P100 amplitude for each check size within each age group.
# =========================

def descriptive_by_age_group(df):
    outcomes = [
        "P100_Latency_12x16_ms",
        "N75_P100_Amplitude_12x16_uV",
        "P100_Latency_48x64_ms",
        "N75_P100_Amplitude_48x64_uV",
    ]

    age_order = ["16-25", "26-40", "41-60", "61+"]

    rows = []

    for group in age_order:
        group_df = df[df["Age_Group"] == group]

        row = {
            "Age_Group": group,
            "N": len(group_df)
        }

        for outcome in outcomes:
            values = group_df[outcome].dropna()
            mean_value = values.mean()
            sd_value = values.std(ddof=1)

            row[f"{outcome}_Mean"] = mean_value
            row[f"{outcome}_SD"] = sd_value
            row[f"{outcome}_Mean_SD"] = f"{mean_value:.2f} +/- {sd_value:.2f}"

        rows.append(row)

    return pd.DataFrame(rows)

# =========================
# SECONDARY FACTOR CORRELATIONS: This section assesses the relationship between height, weight, and head size and each
# VEP measurement by calculating the Pearson correlation coefficient (r) and corresponding p-value.
# =========================

def secondary_factor_correlations(df):
    factors = [HEIGHT, WEIGHT, HEAD_SIZE]

    outcomes = [
        "P100_Latency_12x16_ms",
        "N75_P100_Amplitude_12x16_uV",
        "P100_Latency_48x64_ms",
        "N75_P100_Amplitude_48x64_uV",
    ]

    rows = []

    for factor in factors:
        for outcome in outcomes:
            sub = df[[factor, outcome]].dropna()

            r_value, p_value = stats.pearsonr(
                sub[factor],
                sub[outcome]
            )

            rows.append({
                "Secondary_Factor": factor,
                "VEP_Outcome": outcome,
                "N": len(sub),
                "Pearson_r": r_value,
                "p_value": p_value,
                "Statistically_Significant": "Yes" if p_value < 0.05 else "No"
            })

    return pd.DataFrame(rows)

# =========================
# SEX ANALYSIS: This section compares the P100 latency and N75-P100 amplitude between male and female participants using
# Welch's independent-samples t-test, reporting the mean +/- SD and corresponding p-value for each VEP measurement.
# =========================

def sex_analysis(df):
    outcomes = [
        "P100_Latency_12x16_ms",
        "N75_P100_Amplitude_12x16_uV",
        "P100_Latency_48x64_ms",
        "N75_P100_Amplitude_48x64_uV",
    ]

    rows = []

    sex_clean = df[SEX].astype(str).str.strip().str.lower()

    for outcome in outcomes:
        female = df.loc[sex_clean == "female", outcome].dropna()
        male = df.loc[sex_clean == "male", outcome].dropna()

        if len(female) < 2 or len(male) < 2:
            continue

        t_value, p_value = stats.ttest_ind(
            female,
            male,
            equal_var=False
        )

        rows.append({
            "VEP_Outcome": outcome,
            "Female_N": len(female),
            "Female_Mean": female.mean(),
            "Female_SD": female.std(ddof=1),
            "Female_Mean_SD": f"{female.mean():.2f} +/- {female.std(ddof=1):.2f}",
            "Male_N": len(male),
            "Male_Mean": male.mean(),
            "Male_SD": male.std(ddof=1),
            "Male_Mean_SD": f"{male.mean():.2f} +/- {male.std(ddof=1):.2f}",
            "t_value": t_value,
            "p_value": p_value,
            "Statistically_Significant": "Yes" if p_value < 0.05 else "No"
        })

    return pd.DataFrame(rows)

# =========================
# MAIN: This section executes the complete analysis workflow by loading the prepared dataset, performing the age,
# descriptive, secondary-factor, and sex analyses, generating the required graphs, and saving all statistical results
# to CSV and Excel files.
# =========================

def main():
    print("Simplified VEP Statistical Analysis")
    print("=" * 70)

    df = load_data()

    print(f"Participants: {len(df)}")
    print(f"Age range: {df[AGE].min():.0f}-{df[AGE].max():.0f} years")

    age_results = []

    age_results.append(
        create_age_graph(
            df,
            "N75_P100_Amplitude_12x16_uV",
            "N75-P100 amplitude (uV)",
            "Age_vs_Amplitude_12x16.png"
        )
    )

    age_results.append(
        create_age_graph(
            df,
            "N75_P100_Amplitude_48x64_uV",
            "N75-P100 amplitude (uV)",
            "Age_vs_Amplitude_48x64.png"
        )
    )

    age_results.append(
        create_age_graph(
            df,
            "P100_Latency_12x16_ms",
            "P100 latency (ms)",
            "Age_vs_Latency_12x16.png"
        )
    )

    age_results.append(
        create_age_graph(
            df,
            "P100_Latency_48x64_ms",
            "P100 latency (ms)",
            "Age_vs_Latency_48x64.png"
        )
    )

    age_results_df = pd.DataFrame(age_results)
    descriptive_df = descriptive_by_age_group(df)
    secondary_df = secondary_factor_correlations(df)
    sex_df = sex_analysis(df)

    age_results_df.to_csv(
        OUTPUT_DIR / "Age_Correlation_Results.csv",
        index=False
    )

    descriptive_df.to_csv(
        OUTPUT_DIR / "Descriptive_By_Age_Group.csv",
        index=False
    )

    secondary_df.to_csv(
        OUTPUT_DIR / "Secondary_Factor_Correlations.csv",
        index=False
    )

    sex_df.to_csv(
        OUTPUT_DIR / "Sex_Analysis.csv",
        index=False
    )

    excel_output = OUTPUT_DIR / "VEP_Simplified_Statistical_Results.xlsx"

    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        age_results_df.to_excel(
            writer,
            sheet_name="Age Correlations",
            index=False
        )
        descriptive_df.to_excel(
            writer,
            sheet_name="Age Group Descriptives",
            index=False
        )
        secondary_df.to_excel(
            writer,
            sheet_name="Secondary Correlations",
            index=False
        )
        sex_df.to_excel(
            writer,
            sheet_name="Sex Analysis",
            index=False
        )

    print("\nAGE CORRELATION RESULTS")
    print("=" * 70)

    for _, row in age_results_df.iterrows():
        print(f"\n{row['Outcome']}")
        print(f"r       = {row['Pearson_r']:.3f}")
        print(f"p-value = {row['p_value']:.6f}")
        print(f"R^2     = {row['R_squared']:.3f}")
        print(f"Significant = {row['Statistically_Significant']}")

    print("\nAnalysis complete.")
    print(f"Results folder: {OUTPUT_DIR}")
    print(f"Graphs folder: {FIGURE_DIR}")
    print(f"Excel results: {excel_output}")


if __name__ == "__main__":
    main()