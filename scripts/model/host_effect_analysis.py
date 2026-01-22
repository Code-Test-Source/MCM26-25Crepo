from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def load_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "athletes_cleaned_FRA.csv"
    df = pd.read_csv(data_path)
    return df


def add_new_event_flag(df: pd.DataFrame) -> pd.DataFrame:
    first_year_by_event = df.groupby("Event")['Year'].transform("min")
    df = df.copy()
    df["is_new_event"] = df["Year"] == first_year_by_event
    return df


def summarize_group(df: pd.DataFrame, label: str) -> Dict:
    n = len(df)
    medal_n = int(df["Is_Medal"].sum())
    medal_rate = medal_n / n if n else np.nan
    return {
        "label": label,
        "n": n,
        "medal_n": medal_n,
        "medal_rate": medal_rate,
        "series": df["Is_Medal"].values,
    }


def chi_square_test(group_a: Dict, group_b: Dict) -> Optional[Dict]:
    if group_a["n"] == 0 or group_b["n"] == 0:
        return None

    contingency = np.array([
        [group_a["medal_n"], group_a["n"] - group_a["medal_n"]],
        [group_b["medal_n"], group_b["n"] - group_b["medal_n"]],
    ])

    chi2, p_value, dof, expected = chi2_contingency(contingency)
    return {
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "expected": expected,
    }


def plot_medal_rates(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary_df["label"], summary_df["medal_rate"], color=["#2c7bb6", "#abd9e9", "#fdae61"])
    ax.set_ylabel("Medal rate")
    ax.set_ylim(0, 1)
    ax.set_title("Host effect: medal rates by group")
    for idx, row in summary_df.iterrows():
        ax.text(idx, row["medal_rate"] + 0.01, f"{row['medal_rate']:.2%}" if not np.isnan(row['medal_rate']) else "NA", ha="center")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    df = load_data()
    df = add_new_event_flag(df)

    host_rows = df[df["is_host"] == 1][["NOC", "Year", "City"]].drop_duplicates().sort_values(["Year", "NOC"])
    host_nocs = host_rows["NOC"].unique()

    df_hosts_only = df[df["NOC"].isin(host_nocs)].copy()

    host_new = summarize_group(
        df_hosts_only[(df_hosts_only["is_host"] == 1) & (df_hosts_only["is_new_event"])],
        "Host year + new events",
    )
    host_existing = summarize_group(
        df_hosts_only[(df_hosts_only["is_host"] == 1) & (~df_hosts_only["is_new_event"])],
        "Host year + existing events",
    )
    nonhost_existing = summarize_group(
        df_hosts_only[(df_hosts_only["is_host"] == 0) & (~df_hosts_only["is_new_event"])],
        "Non-host year + existing events",
    )

    summary_df = pd.DataFrame(
        [host_new, host_existing, nonhost_existing],
        columns=["label", "n", "medal_n", "medal_rate", "series"],
    )

    print("Host years (NOC, Year, City):")
    print(host_rows.to_string(index=False))
    print("\nMedal rate summary:")
    print(summary_df.drop(columns=["series"]))

    test1 = chi_square_test(host_existing, nonhost_existing)
    test2 = chi_square_test(host_new, host_existing)

    print("\nTest 1: Host existing vs Non-host existing (is_host effect)")
    if test1:
        print(f"chi2={test1['chi2']:.3f}, p={test1['p_value']:.4f}, dof={test1['dof']}")
    else:
        print("Insufficient data for test 1")

    print("\nTest 2: Host new vs Host existing (new event effect)")
    if test2:
        print(f"chi2={test2['chi2']:.3f}, p={test2['p_value']:.4f}, dof={test2['dof']}")
    else:
        print("Insufficient data for test 2")

    output_path = Path(__file__).resolve().parent / "host_effect_medal_rates.png"
    plot_medal_rates(summary_df, output_path)
    print(f"\nSaved medal rate chart to {output_path}")


if __name__ == "__main__":
    main()