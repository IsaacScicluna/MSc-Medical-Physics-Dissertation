"""
SSEP Statistical Analysis
====================================

This script analyses the SSEP dataset contained in:

    SSEP Data.xlsx

The Excel file should be placed in the SAME folder as this Python script.

Main analyses
-------------
1. Age vs mean N20 latency
   - Pearson r
   - p-value
   - R^2
   - black scatter markers and solid black regression line

2. Age vs mean N20-P25 amplitude
   - Pearson r
   - p-value
   - R^2
   - black scatter markers and solid black regression line

3. Descriptive statistics by age group
   - N
   - Mean
   - Standard deviation (SD)
   - Mean +/- SD

4. Secondary-factor correlations
   - Arm length
   - Height
   - Weight
   against:
       a. Mean N20 latency
       b. Mean N20-P25 amplitude

5. Sex analysis
   - Female vs male mean +/- SD
   - Welch's independent-samples t-test
   - t-value and p-value

The LEFT and RIGHT SSEP measurements are averaged for each participant so that
each participant contributes one N20 latency and one N20-P25 amplitude value.

Age groups used:
    16-40 years
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
# SETTINGS: This section defines the input SSEP dataset and establishes the locations where the generated statistical
# results and figures will be stored.
# =============================================================================

# The Excel file with the SSEP data is expected to be in the same folder as this Python script.
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "SSEP Data.xlsx"

OUTPUT_DIR = SCRIPT_DIR / "SSEP_Statistical_Analysis"
FIGURE_DIR = OUTPUT_DIR / "Figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
DPI = 500


# =============================================================================
# COLUMN NAMES: This section defines the variable names corresponding to the demographic, anthropometric, N20 latency,
# and N20-P25 amplitude data contained within the SSEP dataset.
# =============================================================================

VOLUNTEER = "volunteer"
AGE = "Age"
SEX = "Sex"

ARM_LENGTH = "Arm Length/cm"
HEIGHT = "Height/cm"
WEIGHT = "Weight/kg"

LEFT_N20_LATENCY = "Left N20 Latency/ms"
LEFT_N20_P25_AMPLITUDE = "Left N20-P25 Amplitude/uv"

RIGHT_N20_LATENCY = "Right N20 Latency/ms"
RIGHT_N20_P25_AMPLITUDE = "Right N20-P25 Amplitude/uV"


# =============================================================================
# DATA PREPARATION: This section loads the SSEP dataset, converts the relevant variables to numerical format, calculates
# the mean left-right N20 latency and N20-P25 amplitude for each participant, and assigns each participant to the
# appropriate age group.
# =============================================================================

def load_data():
    """
    Loading the SSEP Excel dataset, converting numerical columns to numeric format,
    calculating the bilateral mean SSEP measurements, and assigning age groups.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find the Excel file:\n{INPUT_FILE}\n\n"
            "Place 'SSEP Data.xlsx' in the same folder as this Python script."
        )

    df = pd.read_excel(INPUT_FILE)

    required_columns = [
        VOLUNTEER,
        AGE,
        SEX,
        ARM_LENGTH,
        HEIGHT,
        WEIGHT,
        LEFT_N20_LATENCY,
        LEFT_N20_P25_AMPLITUDE,
        RIGHT_N20_LATENCY,
        RIGHT_N20_P25_AMPLITUDE,
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
        ARM_LENGTH,
        HEIGHT,
        WEIGHT,
        LEFT_N20_LATENCY,
        LEFT_N20_P25_AMPLITUDE,
        RIGHT_N20_LATENCY,
        RIGHT_N20_P25_AMPLITUDE,
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------------------------------------------------------
    # Bilateral mean values
    # -------------------------------------------------------------------------

    df["Mean_N20_Latency_ms"] = df[
        [LEFT_N20_LATENCY, RIGHT_N20_LATENCY]
    ].mean(axis=1)

    df["Mean_N20_P25_Amplitude_uV"] = df[
        [LEFT_N20_P25_AMPLITUDE, RIGHT_N20_P25_AMPLITUDE]
    ].mean(axis=1)

    # -------------------------------------------------------------------------
    # Age groups
    # -------------------------------------------------------------------------

    def assign_age_group(age):
        if 16 <= age <= 40:
            return "16-40"
        elif 41 <= age <= 60:
            return "41-60"
        elif age >= 61:
            return "61+"
        return np.nan

    df["Age_Group"] = df[AGE].apply(assign_age_group)

    return df


# =============================================================================
# AGE CORRELATION GRAPHS: This section generates scatter plots with linear regression lines to assess the relationship
# between age and the mean N20 latency and N20-P25 amplitude, while calculating the corresponding Pearson correlation
# coefficient (r), p-value and coefficient of determination (r-squared).
# =============================================================================

def create_age_graph(df, outcome, ylabel, filename):
    """
    Creating an age-vs-SSEP scatter plot and calculating:
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

    # Pearson correlation.
    r_value, p_value = stats.pearsonr(x, y)

    # Simple linear regression used for the plotted trend line.
    slope, intercept, _, _, _ = stats.linregress(x, y)

    # For simple linear regression with one predictor:
    # R^2 = Pearson r squared.
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

    ax.text(
        0.05,
        0.90,
        statistics_text,
        transform=ax.transAxes,
        verticalalignment="top",
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
# DESCRIPTIVE STATISTICS BY AGE GROUP: This section calculates the mean and standard deviation (SD) of the N20 latency
# and N20-P25 amplitude within each age group.
# =============================================================================

def descriptive_by_age_group(df):
    """
    Calculating N, mean, SD, and mean +/- SD for N20 latency and N20-P25 amplitude
    within each age group.
    """

    outcomes = [
        "Mean_N20_Latency_ms",
        "Mean_N20_P25_Amplitude_uV",
    ]

    age_order = [
        "16-40",
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
# SECONDARY-FACTOR CORRELATIONS: This section assesses the relationship between arm length, height, and weight and the
# mean N20 latency and N20-P25 amplitude by calculating the Pearson correlation coefficient (r) and corresponding
# p-value.
# =============================================================================

def secondary_factor_correlations(df):
    """
    Assessing correlations between arm length, height, weight and the SSEP outcomes
    using Pearson's correlation coefficient.
    """

    factors = [
        ARM_LENGTH,
        HEIGHT,
        WEIGHT,
    ]

    outcomes = [
        "Mean_N20_Latency_ms",
        "Mean_N20_P25_Amplitude_uV",
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
                "SSEP_Outcome": outcome,
                "N": len(sub),
                "Pearson_r": r_value,
                "p_value": p_value,
                "Statistically_Significant": (
                    "Yes" if p_value < ALPHA else "No"
                ),
            })

    return pd.DataFrame(rows)


# =============================================================================
# SEX ANALYSIS: This section compares the mean N20 latency and N20-P25 amplitude between male and female participants
# using Welch's t-test, reporting the mean +/- SD and corresponding p-value for each SSEP measurement.
# =============================================================================

def sex_analysis(df):
    """
    Comparing male and female participants using Welch's t-test.

    Welch's test is used because it does not require equal population
    variances and remains suitable when group sizes differ.

    Because the SSEP sample is small, this should be interpreted as a
    secondary/exploratory analysis.
    """

    outcomes = [
        "Mean_N20_Latency_ms",
        "Mean_N20_P25_Amplitude_uV",
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
            "SSEP_Outcome": outcome,

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
# maximum for the main SSEP measurements and relevant demographic and anthropometric variables across all participants.
# =============================================================================

def overall_descriptive_statistics(df):
    """
    Calculating the overall descriptive statistics for the main SSEP outcomes and
    anthropometric variables.
    """

    variables = [
        AGE,
        ARM_LENGTH,
        HEIGHT,
        WEIGHT,
        "Mean_N20_Latency_ms",
        "Mean_N20_P25_Amplitude_uV",
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
# MAIN: This section executes the complete SSEP analysis workflow by loading and preparing the dataset, performing the
# age, descriptive, secondary-factor, and sex analyses, generating the required graphs, and saving all statistical
# results to CSV and Excel files.
# =============================================================================

def main():

    print("SSEP Statistical Analysis")
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
        .reindex(["16-40", "41-60", "61+"])
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
            outcome="Mean_N20_Latency_ms",
            ylabel="N20 latency (ms)",
            filename="Age_vs_N20_Latency.png"
        )
    )

    age_results.append(
        create_age_graph(
            df=df,
            outcome="Mean_N20_P25_Amplitude_uV",
            ylabel="N20-P25 amplitude (uV)",
            filename="Age_vs_N20_P25_Amplitude.png"
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

    prepared_data_path = OUTPUT_DIR / "Prepared_SSEP_Data.csv"
    age_results_path = OUTPUT_DIR / "Age_Correlation_Results.csv"
    overall_descriptive_path = OUTPUT_DIR / "Overall_Descriptive_Statistics.csv"
    age_group_descriptive_path = OUTPUT_DIR / "Descriptive_By_Age_Group.csv"
    secondary_path = OUTPUT_DIR / "Secondary_Factor_Correlations.csv"
    sex_path = OUTPUT_DIR / "Sex_Analysis.csv"

    df.to_csv(
        prepared_data_path,
        index=False
    )

    age_results_df.to_csv(
        age_results_path,
        index=False
    )

    overall_descriptive_df.to_csv(
        overall_descriptive_path,
        index=False
    )

    age_group_descriptive_df.to_csv(
        age_group_descriptive_path,
        index=False
    )

    secondary_df.to_csv(
        secondary_path,
        index=False
    )

    sex_df.to_csv(
        sex_path,
        index=False
    )

    # -------------------------------------------------------------------------
    # SAVE COMBINED EXCEL WORKBOOK
    # -------------------------------------------------------------------------

    excel_output = (
        OUTPUT_DIR /
        "SSEP_Statistical_Analysis_Results.xlsx"
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