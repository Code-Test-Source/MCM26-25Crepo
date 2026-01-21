import csv
import statistics
from pathlib import Path

from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'outputs' / 'processed_data'
INPUT = DATA_DIR / "china_medals_by_year_sport.csv"
OUTPUT = DATA_DIR / "china_sport_stability_cv.csv"


def to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def compute_cv(values):
    # 两点或多点的变异系数；用于相邻年份窗口
    if not values:
        return None
    mean_v = statistics.mean(values)
    if mean_v == 0:
        return None
    if len(values) == 1:
        return 0.0
    sd = statistics.pstdev(values)
    return sd / mean_v if mean_v else None


def main():
    # 改为“相邻年份”的稳定性指标：对每个 (Sport, Year>=prevYear) 计算基于 [prev, curr] 的两点CV
    # 输出到逐年层级，便于后续与其它年度指标联动。
    by_sport_year = {}  # sport -> {year: weighted321}
    with INPUT.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        for row in reader:
            sport = row.get("Sport", "")
            year = row.get("Year", "")
            w = to_int(row.get("Weighted321"))
            if not sport or not year:
                continue
            m = by_sport_year.setdefault(sport, {})
            m[year] = m.get(year, 0) + w

    # 生成逐年CV（使用相邻两年）
    records = []
    for sport, ymap in by_sport_year.items():
        years_sorted = sorted(ymap.keys())
        prev_year = None
        for y in years_sorted:
            curr = ymap[y]
            if prev_year is None:
                # 第一年的相邻CV无法计算，留空
                records.append(
                    {
                        "Sport": sport,
                        "Year": y,
                        "PrevYear": "",
                        "PrevWeighted321": "",
                        "CurrWeighted321": curr,
                        "CV2YR": "",
                    }
                )
            else:
                prev = ymap.get(prev_year, 0)
                cv = compute_cv([prev, curr]) if (prev or curr) else None
                records.append(
                    {
                        "Sport": sport,
                        "Year": y,
                        "PrevYear": prev_year,
                        "PrevWeighted321": prev,
                        "CurrWeighted321": curr,
                        "CV2YR": round(cv, 5) if cv is not None else "",
                    }
                )
            prev_year = y

    # 按年内对 CV2YR 做 z 分数，便于横向比较（可选）
    # 计算每年所有运动的 CV2YR 的均值与方差
    by_year_vals = {}
    for r in records:
        y = r["Year"]
        v = r["CV2YR"]
        if isinstance(v, float):
            by_year_vals.setdefault(y, []).append(v)

    year_stats = {}
    for y, vals in by_year_vals.items():
        if vals:
            mean_v = statistics.mean(vals)
            sd_v = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        else:
            mean_v, sd_v = 0.0, 0.0
        year_stats[y] = (mean_v, sd_v)

    for r in records:
        y = r["Year"]
        v = r["CV2YR"]
        mean_v, sd_v = year_stats.get(y, (0.0, 0.0))
        if isinstance(v, float) and sd_v > 0:
            r["CV2YR_Z"] = round((v - mean_v) / sd_v, 4)
        elif isinstance(v, float):
            r["CV2YR_Z"] = 0.0
        else:
            r["CV2YR_Z"] = ""

    # 排序：Year, Sport
    records.sort(key=lambda r: (r["Year"], r["Sport"]))

    fieldnames = [
        "Year",
        "Sport",
        "PrevYear",
        "PrevWeighted321",
        "CurrWeighted321",
        "CV2YR",
        "CV2YR_Z",
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"Wrote {len(records)} year-sport adjacent-year CV records to {OUTPUT}")


if __name__ == "__main__":
    main()
