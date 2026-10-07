"""
BAEP Statistical Analysis
====================================

This script analyses the BAEP dataset contained in:

    BAEP Data.xlsx

The Excel file should be placed in the SAME folder as this Python script.

Main analyses
-------------
1. Age vs mean Wave I latency
   - Pearson r
   - p-value
   - R^

2. Age vs mean Wave V latency
   - Pearson r
   - p-value
   - R^2

3. Age vs mean Wave I amplitude
   - Pearson r
   - p-value
   - R^2

4. Age vs mean Wave V amplitude
   - Pearson r
   - p-value
   - R^2

5. Descriptive statistics by age group
   - N
   - Mean
   - Standard deviation (SD)
   - Mean +/- SD

6. Secondary-factor correlations
   - Head size
   - Height
   - Weight
   against all four BAEP measures

7. Sex analysis
   - Female vs male mean +/- SD
   - Welch's independent-samples t-test
   - t-value and p-value

The LEFT and RIGHT ear measurements are averaged for each participant so that
each participant contributes one value for each BAEP measure.

Age groups used:
    20-40 years
    41-60 years
    61+ years

Required packages
-----------------
pandas
numpy
scipy
matplotlib
openpyxl

Install if required with:
    python -m pip install pandas numpy scipy matplotlib openpyxl
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# =============================================================================
# SETTINGS: This section defines the input BAEP dataset and establishes the locations where the generated statistical
# results and figures will be stored.
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "BAEP Data.xlsx"

OUTPUT_DIR = SCRIPT_DIR / "BAEP_Statistical_Analysis"
FIGURE_DIR = OUTPUT_DIR / "Figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
DPI = 500


# =============================================================================
# COLUMN NAMES: This section defines the variable names corresponding to the demographic, anthropometric, and bilateral
# Wave I and Wave V latency and amplitude data contained within the BAEP dataset.
# =============================================================================

VOLUNTEER = "volunteer"
AGE = "Age"
SEX = "Sex"

HEAD_SIZE = "Head Size/cm"
HEIGHT = "Height/cm"
WEIGHT = "Weight/kg"

LEFT_WAVE_I_LATENCY = "Left Wave I Latency/ms"
LEFT_WAVE_I_AMPLITUDE = "Left Wave I Amplitude/uv"
LEFT_WAVE_V_LATENCY = "Left Wave V Latency/ms"
LEFT_WAVE_V_AMPLITUDE = "Left Wave V Amplitude/uv"

RIGHT_WAVE_I_LATENCY = "Right Wave I Latency/ms"
RIGHT_WAVE_I_AMPLITUDE = "Right Wave I Amplitude/uv"
RIGHT_WAVE_V_LATENCY = "Right Wave V Latency/ms"
RIGHT_WAVE_V_AMPLITUDE = "Right Wave V Amplitude/uv"


# =============================================================================
# DATA PREPARATION: This section loads the BAEP dataset, converts the relevant variables to numerical format, calculates
# the mean left-right Wave I and Wave V latency and amplitude for each participant, and assigns each participant to the
# respective age group.
# =============================================================================

def load_data():
    """
    Loading the BAEP Excel dataset, converting numerical columns to numeric format,
    calculating the bilateral mean Wave I and Wave V measurements, and assigning age groups.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find the Excel file:\n{INPUT_FILE}\n\n"
            "Place 'BAEP Data.xlsx' in the same folder as this Python script."
        )

    df = pd.read_excel(INPUT_FILE)

    required_columns = [
        VOLUNTEER,
        AGE,
        SEX,
        HEAD_SIZE,
        HEIGHT,
        WEIGHT,
        LEFT_WAVE_I_LATENCY,
        LEFT_WAVE_I_AMPLITUDE,
        LEFT_WAVE_V_LATENCY,
        LEFT_WAVE_V_AMPLITUDE,
        RIGHT_WAVE_I_LATENCY,
        RIGHT_WAVE_I_AMPLITUDE,
        RIGHT_WAVE_V_LATENCY,
        RIGHT_WAVE_V_AMPLITUDE,
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing from the Excel file:\n"
            + "\n".join(missing_columns)
        )

    numeric_columns = [
        AGE,
        HEAD_SIZE,
        HEIGHT,
        WEIGHT,
        LEFT_WAVE_I_LATENCY,
        LEFT_WAVE_I_AMPLITUDE,
        LEFT_WAVE_V_LATENCY,
        LEFT_WAVE_V_AMPLITUDE,
        RIGHT_WAVE_I_LATENCY,
        RIGHT_WAVE_I_AMPLITUDE,
        RIGHT_WAVE_V_LATENCY,
        RIGHT_WAVE_V_AMPLITUDE,
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Bilateral mean values.
    df["Mean_Wave_I_Latency_ms"] = df[
        [LEFT_WAVE_I_LATENCY, RIGHT_WAVE_I_LATENCY]
    ].mean(axis=1)

    df["Mean_Wave_I_Amplitude_uV"] = df[
        [LEFT_WAVE_I_AMPLITUDE, RIGHT_WAVE_I_AMPLITUDE]
    ].mean(axis=1)

    df["Mean_Wave_V_Latency_ms"] = df[
        [LEFT_WAVE_V_LATENCY, RIGHT_WAVE_V_LATENCY]
    ].mean(axis=1)

    df["Mean_Wave_V_Amplitude_uV"] = df[
        [LEFT_WAVE_V_AMPLITUDE, RIGHT_WAVE_V_AMPLITUDE]
    ].mean(axis=1)

    def assign_age_group(age):
        if 20 <= age <= 40:
            return "20-40"
        elif 41 <= age <= 60:
            return "41-60"
        elif age >= 61:
            return "61+"
        return np.nan

    df["Age_Group"] = df[AGE].apply(assign_age_group)

    return df


# =============================================================================
# AGE CORRELATION GRAPHS: This section generates scatter plots with linear regression lines to assess the relationship
# between age and the mean Wave I and Wave V latencies and amplitudes, while calculating the corresponding Pearson
# correlation coefficient (r), p-value, and coefficient of determination (R-squared).
# =============================================================================

def create_age_graph(df, outcome, ylabel, filename):
    """
    Creating an age-vs-BAEP scatter plot and calculating:
        - Pearson correlation coefficient (r)
        - p-value
        - coefficient of determination (R^2)

    A simple linear regression line is added to the graph.
    """

    sub = df[[AGE, outcome]].dropna()

    x = sub[AGE].to_numpy()
    y = sub[outcome].to_numpy()

    if len(sub) < 3:
        raise ValueError(
            f"Not enough complete observations to analyse {outcome}."
        )

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
    ax.grid(False)

    if p_value < 0.001:
        p_text = "$p$ < 0.001"
    else:
        p_text = f"$p$ = {p_value:.3f}"

    statistics_text = (
        f"$r$ = {r_value:.3f}\n"
        f"{p_text}\n"
        f"$R^2$ = {r_squared:.3f}"
    )

    # Position statistics according to measurement type.
    if "Amplitude" in outcome:
        text_x = 0.05
        text_y = 0.05
        vertical_alignment = "bottom"
    else:  # Latency
        text_x = 0.05
        text_y = 0.95
        vertical_alignment = "top"

    ax.text(
        text_x,
        text_y,
        statistics_text,
        transform=ax.transAxes,
        verticalalignment=vertical_alignment,
        horizontalalignment="left",
        fontsize=10
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / filename,
        dpi=DPI,
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
        "Statistically_Significant": (
            "Yes" if p_value < ALPHA else "No"
        ),
    }


# =============================================================================
# DESCRIPTIVE STATISTICS BY AGE GROUP: This section calculates the mean and standard deviation (SD) of the Wave I and
# Wave V latencies and amplitudes within each age group.
# =============================================================================

def descriptive_by_age_group(df):
    """
    Calculating N, mean, SD, and mean +/- SD for the four BAEP measures
    within each age group.
    """

    outcomes = [
        "Mean_Wave_I_Latency_ms",
        "Mean_Wave_I_Amplitude_uV",
        "Mean_Wave_V_Latency_ms",
        "Mean_Wave_V_Amplitude_uV",
    ]

    age_order = [
        "20-40",
        "41-60",
        "61+",
    ]

    rows = []

    for group in age_order:
        group_df = df[df["Age_Group"] == group]

        row = {
            "Age_Group": group,
            "N": len(group_df),
        }

        for outcome in outcomes:
            values = group_df[outcome].dropna()

            mean_value = values.mean()
            sd_value = values.std(ddof=1)

            row[f"{outcome}_Mean"] = mean_value
            row[f"{outcome}_SD"] = sd_value

            if len(values) >= 2:
                row[f"{outcome}_Mean_SD"] = (
                    f"{mean_value:.2f} +/- {sd_value:.2f}"
                )
            elif len(values) == 1:
                row[f"{outcome}_Mean_SD"] = (
                    f"{mean_value:.2f} +/- N/A"
                )
            else:
                row[f"{outcome}_Mean_SD"] = ""

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# SECONDARY-FACTOR CORRELATIONS: This section assesses the relationship between head size, height, and weight and the
# mean Wave I and Wave V latencies and amplitudes by calculating the Pearson correlation coefficient (r) and
# corresponding p-value.
# =============================================================================

def secondary_factor_correlations(df):
    """
    Assessing the correlations between head size, height, weight and the BAEP measures
    using Pearson's correlation coefficient.
    """

    factors = [
        HEAD_SIZE,
        HEIGHT,
        WEIGHT,
    ]

    outcomes = [
        "Mean_Wave_I_Latency_ms",
        "Mean_Wave_I_Amplitude_uV",
        "Mean_Wave_V_Latency_ms",
        "Mean_Wave_V_Amplitude_uV",
    ]

    rows = []

    for factor in factors:
        for outcome in outcomes:

            sub = df[[factor, outcome]].dropna()

            if len(sub) < 3:
                continue

            r_value, p_value = stats.pearsonr(
                sub[factor],
                sub[outcome]
            )

            rows.append({
                "Secondary_Factor": factor,
                "BAEP_Outcome": outcome,
                "N": len(sub),
                "Pearson_r": r_value,
                "p_value": p_value,
                "Statistically_Significant": (
                    "Yes" if p_value < ALPHA else "No"
                ),
            })

    return pd.DataFrame(rows)


# =============================================================================
# SEX ANALYSIS: This section compare the mean Wave I and Wave V latencies and amplitudes between male and female
# participants using Welch's t-test, reporting the mean +/- SD and corresponding p-value for each BAEP measurement.
# =============================================================================

def sex_analysis(df):
    """
    Comparing the male and female participants using Welch's t-test.

    Because the BAEP sample is small, this is treated as a secondary/exploratory
    analysis.
    """

    outcomes = [
        "Mean_Wave_I_Latency_ms",
        "Mean_Wave_I_Amplitude_uV",
        "Mean_Wave_V_Latency_ms",
        "Mean_Wave_V_Amplitude_uV",
    ]

    rows = []

    sex_clean = (
        df[SEX]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    for outcome in outcomes:

        female = df.loc[
            sex_clean == "female",
            outcome
        ].dropna()

        male = df.loc[
            sex_clean == "male",
            outcome
        ].dropna()

        if len(female) < 2 or len(male) < 2:
            continue

        t_value, p_value = stats.ttest_ind(
            female,
            male,
            equal_var=False
        )

        rows.append({
            "BAEP_Outcome": outcome,

            "Female_N": len(female),
            "Female_Mean": female.mean(),
            "Female_SD": female.std(ddof=1),
            "Female_Mean_SD": (
                f"{female.mean():.2f} +/- "
                f"{female.std(ddof=1):.2f}"
            ),

            "Male_N": len(male),
            "Male_Mean": male.mean(),
            "Male_SD": male.std(ddof=1),
            "Male_Mean_SD": (
                f"{male.mean():.2f} +/- "
                f"{male.std(ddof=1):.2f}"
            ),

            "t_value": t_value,
            "p_value": p_value,

            "Statistically_Significant": (
                "Yes" if p_value < ALPHA else "No"
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# OVERALL DESCRIPTIVE STATISTICS: This section calculates the overall mean, standard deviation (SD), minimum, and
# maximum for the BAEP measurements and relevant demographic and anthropometric variables across all participants.
# =============================================================================

def overall_descriptive_statistics(df):
    """
    Calculating the overall descriptive statistics for the main BAEP measures and
    demographic/anthropometric variables.
    """

    variables = [
        AGE,
        HEAD_SIZE,
        HEIGHT,
        WEIGHT,
        "Mean_Wave_I_Latency_ms",
        "Mean_Wave_I_Amplitude_uV",
        "Mean_Wave_V_Latency_ms",
        "Mean_Wave_V_Amplitude_uV",
    ]

    rows = []

    for variable in variables:

        values = df[variable].dropna()

        rows.append({
            "Variable": variable,
            "N": len(values),
            "Mean": values.mean(),
            "SD": values.std(ddof=1),
            "Minimum": values.min(),
            "Maximum": values.max(),
            "Mean_SD": (
                f"{values.mean():.2f} +/- "
                f"{values.std(ddof=1):.2f}"
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN: This section executes the complete BAEP analysis workflow by loading and preparing the dataset, performing the
# age, descriptive, secondary-factor, and sex analyses, generating the required graphs, and saving all statistical
# results to CSV and Excel files.
# =============================================================================

def main():

    print("BAEP Statistical Analysis")
    print("=" * 70)

    df = load_data()

    print(f"Participants: {len(df)}")
    print(
        f"Age range: "
        f"{df[AGE].min():.0f}-"
        f"{df[AGE].max():.0f} years"
    )

    print("\nAge-group distribution:")
    print(
        df["Age_Group"]
        .value_counts()
        .reindex(["20-40", "41-60", "61+"])
        .to_string()
    )

    print("\nSex distribution:")
    print(df[SEX].value_counts().to_string())

    # -------------------------------------------------------------------------
    # AGE ANALYSIS
    # -------------------------------------------------------------------------

    age_results = []

    age_results.append(
        create_age_graph(
            df=df,
            outcome="Mean_Wave_I_Latency_ms",
            ylabel="Wave I latency (ms)",
            filename="Age_vs_Wave_I_Latency.png"
        )
    )

    age_results.append(
        create_age_graph(
            df=df,
            outcome="Mean_Wave_I_Amplitude_uV",
            ylabel="Wave I amplitude (uV)",
            filename="Age_vs_Wave_I_Amplitude.png"
        )
    )

    age_results.append(
        create_age_graph(
            df=df,
            outcome="Mean_Wave_V_Latency_ms",
            ylabel="Wave V latency (ms)",
            filename="Age_vs_Wave_V_Latency.png"
        )
    )

    age_results.append(
        create_age_graph(
            df=df,
            outcome="Mean_Wave_V_Amplitude_uV",
            ylabel="Wave V amplitude (uV)",
            filename="Age_vs_Wave_V_Amplitude.png"
        )
    )

    age_results_df = pd.DataFrame(age_results)

    # -------------------------------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # -------------------------------------------------------------------------

    overall_descriptive_df = overall_descriptive_statistics(df)
    age_group_descriptive_df = descriptive_by_age_group(df)

    # -------------------------------------------------------------------------
    # SECONDARY FACTORS
    # -------------------------------------------------------------------------

    secondary_df = secondary_factor_correlations(df)

    # -------------------------------------------------------------------------
    # SEX
    # -------------------------------------------------------------------------

    sex_df = sex_analysis(df)

    # -------------------------------------------------------------------------
    # SAVE CSV OUTPUTS
    # -------------------------------------------------------------------------

    df.to_csv(
        OUTPUT_DIR / "Prepared_BAEP_Data.csv",
        index=False
    )

    age_results_df.to_csv(
        OUTPUT_DIR / "Age_Correlation_Results.csv",
        index=False
    )

    overall_descriptive_df.to_csv(
        OUTPUT_DIR / "Overall_Descriptive_Statistics.csv",
        index=False
    )

    age_group_descriptive_df.to_csv(
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

    # -------------------------------------------------------------------------
    # SAVE COMBINED EXCEL WORKBOOK
    # -------------------------------------------------------------------------

    excel_output = (
        OUTPUT_DIR /
        "BAEP_Statistical_Analysis_Results.xlsx"
    )

    with pd.ExcelWriter(
        excel_output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Prepared Data",
            index=False
        )

        age_results_df.to_excel(
            writer,
            sheet_name="Age Correlations",
            index=False
        )

        overall_descriptive_df.to_excel(
            writer,
            sheet_name="Overall Descriptives",
            index=False
        )

        age_group_descriptive_df.to_excel(
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

    # -------------------------------------------------------------------------
    # PRINT MAIN RESULTS
    # -------------------------------------------------------------------------

    print("\nAGE CORRELATION RESULTS")
    print("=" * 70)

    for _, row in age_results_df.iterrows():

        print(f"\n{row['Outcome']}")
        print(
            f"r       = "
            f"{row['Pearson_r']:.3f}"
        )
        print(
            f"p-value = "
            f"{row['p_value']:.6f}"
        )
        print(
            f"R^2     = "
            f"{row['R_squared']:.3f}"
        )
        print(
            f"Significant = "
            f"{row['Statistically_Significant']}"
        )

    print("\nAnalysis complete.")
    print(f"Results folder: {OUTPUT_DIR}")
    print(f"Graphs folder:  {FIGURE_DIR}")
    print(f"Excel results:  {excel_output}")


if __name__ == "__main__":
    main()