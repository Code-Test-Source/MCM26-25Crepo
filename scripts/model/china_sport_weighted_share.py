import csv
from pathlib import Path

# 输入为分年份分运动的加权统计文件
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'outputs' / 'processed_data'
INPUT = DATA_DIR / "china_medals_by_year_sport.csv"
OUTPUT = DATA_DIR / "china_sport_weighted_share.csv"


def to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def main():
    # 改为逐年计算：对每个年份，计算该年各运动 Weighted321 在该年的占比
    by_year_sport = {}  # (year, sport) -> weighted
    year_totals = {}    # year -> sum weighted

    with INPUT.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        for row in reader:
            year = row.get("Year", "")
            sport = row.get("Sport", "")
            w = to_int(row.get("Weighted321"))
            if not year or not sport:
                continue
            by_year_sport[(year, sport)] = by_year_sport.get((year, sport), 0) + w
            year_totals[year] = year_totals.get(year, 0) + w

    # 生成逐年记录
    records = []
    for (year, sport), w in by_year_sport.items():
        total = year_totals.get(year, 0)
        share = (w / total) if total > 0 else 0.0
        records.append(
            {
                "Year": year,
                "Sport": sport,
                "Weighted321": w,
                "Share": round(share, 6),
                "SharePct": round(share * 100.0, 4),
            }
        )

    # 按年份排序，再按占比降序、运动名升序
    records.sort(key=lambda r: (r["Year"], -r["Share"], r["Sport"]))

    # 在每个年份内添加排名
    current_year = None
    rank = 0
    for r in records:
        if r["Year"] != current_year:
            current_year = r["Year"]
            rank = 1
        else:
            rank += 1
        r["RankWithinYear"] = rank

    fieldnames = [
        "Year",
        "RankWithinYear",
        "Sport",
        "Weighted321",
        "Share",
        "SharePct",
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"Wrote {len(records)} year-sport share records to {OUTPUT}")


if __name__ == "__main__":
    main()
