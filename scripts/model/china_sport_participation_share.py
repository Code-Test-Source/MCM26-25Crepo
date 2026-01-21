import csv
from pathlib import Path

from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'outputs' / 'processed_data'
RAW_INPUT = _P(ROOT) / '2025_Problem_C_Data' / 'summerOly_athletes.csv'
INPUT = RAW_INPUT if RAW_INPUT.exists() else DATA_DIR / "summerOly_athletes_processed.csv"
OUTPUT = DATA_DIR / "china_sport_participation_share.csv"
MEDAL_INPUT = DATA_DIR / "china_medals_by_year_sport.csv"

# 参与人数口径：按 (Year, Name) 在同一 Sport 去重，
# 这样同一届一个运动中参加多个小项的运动员只计 1 人；跨届会分别计入。


def to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def main():
    # 改为逐年计算参与占比：针对每个 (Year, Sport)，计算中国参与人数 / 世界参与人数
    world_sets = {}  # (year, sport) -> set of names
    china_sets = {}  # (year, sport) -> set of names

    # 读取当年有奖牌的 (Year, Sport) 作为过滤（与原始逻辑一致但更精细到年度）
    medal_year_sport = set()
    if MEDAL_INPUT.exists():
        with MEDAL_INPUT.open(newline="", encoding="utf-8") as medal_src:
            medal_reader = csv.DictReader(medal_src)
            for row in medal_reader:
                sport = row.get("Sport", "")
                year = row.get("Year", "")
                weighted = to_int(row.get("Weighted321"))
                if sport and year and weighted > 0:
                    medal_year_sport.add((year, sport))

    with INPUT.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        for row in reader:
            sport = row.get("Sport", "")
            year = row.get("Year", "")
            name = row.get("Name", "")
            if not sport or not year or not name:
                continue

            key = (year, sport)

            # 世界总体参与者集合（同一届同一运动，同名合并）
            s = world_sets.setdefault(key, set())
            s.add(name)

            # 中国参与者集合（Team=China 或 NOC=CHN）
            if row.get("Team") == "China" or row.get("NOC") == "CHN":
                cs = china_sets.setdefault(key, set())
                cs.add(name)

    # 仅输出当年中国有获奖牌的 (Year, Sport)
    records = []
    all_keys = set(world_sets.keys()) | set(china_sets.keys())
    for (year, sport) in sorted(all_keys):
        if medal_year_sport and (year, sport) not in medal_year_sport:
            continue
        china_cnt = len(china_sets.get((year, sport), set()))
        world_cnt = len(world_sets.get((year, sport), set()))
        share = (china_cnt / world_cnt) if world_cnt > 0 else 0.0
        records.append(
            {
                "Year": year,
                "Sport": sport,
                "ChinaParticipants": china_cnt,
                "WorldParticipants": world_cnt,
                "Share": round(share, 6),
                "SharePct": round(share * 100.0, 4),
            }
        )

    # 按年份、占比排序
    records.sort(key=lambda r: (r["Year"], -r["Share"], r["Sport"]))

    # 每个年份内排名
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
        "ChinaParticipants",
        "WorldParticipants",
        "Share",
        "SharePct",
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"Wrote {len(records)} year-sport participation shares to {OUTPUT}")


if __name__ == "__main__":
    main()
