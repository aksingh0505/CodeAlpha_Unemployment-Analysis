
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save charts without opening GUI windows
import matplotlib.pyplot as plt

# ==========================================
# UNEMPLOYMENT ANALYSIS WITH PYTHON
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_1 = os.path.join(BASE_DIR, "Unemployment in India.csv")
FILE_2 = os.path.join(BASE_DIR, "Unemployment_Rate_upto_11_2020.csv")


def save_chart(filename):
    """Save chart and release memory."""
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=100)
    plt.close()


def load_data(filename):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()

    for col in df.columns:
        if df[col].dtype == object or isinstance(
            df[col].dtype, pd.StringDtype
        ):
            df[col] = df[col].str.strip()

    df = df.rename(columns={
        "Estimated Unemployment Rate (%)": "Unemployment_Rate",
        "Estimated Employed": "Employed",
        "Estimated Labour Participation Rate (%)":
            "Labour_Participation_Rate"
    })

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce", dayfirst=True
    )

    for col in [
        "Unemployment_Rate",
        "Employed",
        "Labour_Participation_Rate"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates()
    df = df.dropna(
        subset=["Date", "Region", "Unemployment_Rate"]
    )
    df = df[df["Unemployment_Rate"].between(0, 100)]

    return df


# 1. LOAD DATA
try:
    df1 = load_data(FILE_1)
    df2 = load_data(FILE_2)
except FileNotFoundError as error:
    print("CSV file not found:", error.filename)
    raise SystemExit(1)

print("\n========== DATASET INFORMATION ==========")
print("Dataset 1 shape:", df1.shape)
print("Dataset 2 shape:", df2.shape)

print("\nMissing values - Dataset 1:")
print(df1.isnull().sum())

print("\nMissing values - Dataset 2:")
print(df2.isnull().sum())


# 2. EXPLORATORY DATA ANALYSIS
print("\n========== STATISTICAL SUMMARY ==========")
print(df1.select_dtypes(include="number").describe())

print("\nDataset 1 average unemployment:",
      round(df1["Unemployment_Rate"].mean(), 2), "%")
print("Dataset 2 average unemployment:",
      round(df2["Unemployment_Rate"].mean(), 2), "%")


# 3. MONTHLY UNEMPLOYMENT TREND
monthly = (
    df2.set_index("Date")["Unemployment_Rate"]
    .resample("MS").mean().dropna()
)

print("\n========== MONTHLY TREND ==========")
print(monthly.round(2))

plt.figure(figsize=(10, 5))
plt.plot(monthly.index, monthly.values, marker="o")
plt.axvline(pd.Timestamp("2020-03-01"),
            linestyle="--", label="COVID-19 period")
plt.title("Monthly Unemployment Rate in India")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.xticks(rotation=45)
save_chart("monthly_unemployment.png")


# 4. COVID-19 IMPACT
df2["Period"] = df2["Date"].apply(
    lambda d: "Pre-COVID"
    if d < pd.Timestamp("2020-03-01")
    else "COVID Period"
)

period_avg = df2.groupby("Period")["Unemployment_Rate"].mean()

print("\n========== COVID-19 IMPACT ==========")
print(period_avg.round(2))

if "Pre-COVID" in period_avg.index and "COVID Period" in period_avg.index:
    before = period_avg["Pre-COVID"]
    during = period_avg["COVID Period"]

    print("Difference:",
          round(during - before, 2), "percentage points")
    print("Relative change:",
          round((during - before) / before * 100, 2), "%")

plt.figure(figsize=(7, 4))
plt.bar(period_avg.index, period_avg.values)
plt.title("Pre-COVID vs COVID-Period Unemployment")
plt.ylabel("Average Unemployment Rate (%)")
save_chart("covid_impact.png")


# 5. REGIONAL ANALYSIS
state_avg = (
    df2.groupby("Region")["Unemployment_Rate"]
    .mean().sort_values(ascending=False)
)

print("\n========== TOP 5 REGIONS ==========")
print(state_avg.head(5).round(2))

print("\n========== BOTTOM 5 REGIONS ==========")
print(state_avg.tail(5).sort_values().round(2))

plt.figure(figsize=(9, 7))
plt.barh(state_avg.sort_values().index,
         state_avg.sort_values().values)
plt.title("Average Unemployment Rate by Region")
plt.xlabel("Unemployment Rate (%)")
save_chart("state_comparison.png")


# 6. RURAL VS URBAN
if "Area" in df1.columns:
    df1["Period"] = df1["Date"].apply(
        lambda d: "Pre-COVID"
        if d < pd.Timestamp("2020-03-01")
        else "COVID Period"
    )

    area_avg = df1.groupby(
        ["Period", "Area"]
    )["Unemployment_Rate"].mean().unstack()

    print("\n========== RURAL VS URBAN ==========")
    print(area_avg.round(2))

    area_avg.plot(kind="bar", figsize=(8, 5))
    plt.title("Rural vs Urban Unemployment")
    plt.ylabel("Unemployment Rate (%)")
    plt.xticks(rotation=0)
    save_chart("rural_urban_comparison.png")


# 7. MONTHLY PATTERNS
df2["Month"] = df2["Date"].dt.month
month_avg = df2.groupby("Month")["Unemployment_Rate"].mean()

month_names = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

month_avg.index = [month_names[m] for m in month_avg.index]

print("\n========== MONTHLY PATTERNS ==========")
print(month_avg.round(2))

plt.figure(figsize=(9, 4))
plt.bar(month_avg.index, month_avg.values)
plt.title("Average Unemployment Rate by Month")
plt.ylabel("Unemployment Rate (%)")
save_chart("monthly_pattern.png")


# 8. LABOUR FORCE PARTICIPATION
if "Labour_Participation_Rate" in df2.columns:
    participation = (
        df2.set_index("Date")["Labour_Participation_Rate"]
        .resample("MS").mean().dropna()
    )

    print("\n========== LABOUR FORCE PARTICIPATION ==========")
    print(participation.round(2))

    plt.figure(figsize=(9, 4))
    plt.plot(participation.index, participation.values, marker="o")
    plt.title("Labour Force Participation Rate")
    plt.xlabel("Month")
    plt.ylabel("Participation Rate (%)")
    plt.xticks(rotation=45)
    save_chart("labour_participation.png")


# 9. EXPORT RESULTS
monthly.rename("Average_Unemployment_Rate").to_csv(
    os.path.join(OUTPUT_DIR, "monthly_unemployment.csv")
)

state_avg.rename("Average_Unemployment_Rate").to_csv(
    os.path.join(OUTPUT_DIR, "state_unemployment_summary.csv")
)

period_avg.rename("Average_Unemployment_Rate").to_csv(
    os.path.join(OUTPUT_DIR, "covid_period_summary.csv")
)

month_avg.rename("Average_Unemployment_Rate").to_csv(
    os.path.join(OUTPUT_DIR, "monthly_pattern_summary.csv")
)

print("\n==========================================")
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print("Charts and CSV summaries saved in:", OUTPUT_DIR)
print("==========================================")
