#!/usr/bin/env python3
"""Generate a draft country_mapping.csv by scanning dataset unique names.

This script uses only the standard library and writes:
  2025_Problem_C_Data/country_mapping.csv
with columns: original_name,suggested_standard,rule_note,year_start,year_end
"""
import csv
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / '2025_Problem_C_Data'
OUT_PATH = DATA_DIR / 'country_mapping.csv'


def normalize_name(s: str) -> str:
    if s is None:
        return ''
    s = s.replace('\u00A0', ' ')
    s = s.strip()
    s = re.sub(r"\s+", ' ', s)
    reps = {
        'U.S.A.': 'United States', 'USA': 'United States', 'U.S.': 'United States',
        'United States of America': 'United States',
        'UK': 'Great Britain', 'U K': 'Great Britain',
        "People's Republic of China": 'China', 'PR China': 'China',
        'Russian Federation': 'Russia', 'ROC': 'Russia',
        'Republic of China': 'Chinese Taipei'
    }
    key = s
    if key in reps:
        return reps[key]
    codes = {'GBR': 'Great Britain', 'USA': 'United States', 'CHN': 'China', 'RUS': 'Russia'}
    if s.upper() in codes:
        return codes[s.upper()]
    if s.isupper() and len(s) <= 3:
        return s
    return s


def collect_unique_names():
    teams = set()
    nocs = set()
    athletes_fp = DATA_DIR / 'summerOly_athletes.csv'
    medals_fp = DATA_DIR / 'summerOly_medal_counts.csv'
    for fp, colset, collector in [(athletes_fp, ['Team'], teams), (medals_fp, ['NOC'], nocs)]:
        if not fp.exists():
            continue
        with fp.open('r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for r in reader:
                for c in colset:
                    v = r.get(c)
                    if v:
                        collector.add(v.strip())
    return teams.union(nocs)


def generate_mapping():
    originals = sorted(collect_unique_names())
    rows = []
    for o in originals:
        std = normalize_name(o)
        note = ''
        if any(x in o.lower() for x in ['soviet', 'yugosl', 'bohemia', 'australasia', 'unified team']):
            note = 'historical_entity'
        rows.append((o, std, note, '', ''))

    with OUT_PATH.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['original_name', 'suggested_standard', 'rule_note', 'year_start', 'year_end'])
        for r in rows:
            w.writerow(r)


if __name__ == '__main__':
    generate_mapping()
    print('Wrote', OUT_PATH)
