#!/usr/bin/env python3
"""Refine the generated country_mapping.csv using heuristic filters.

Outputs: 2025_Problem_C_Data/country_mapping_cleaned.csv
Columns: original_name,suggested_standard,rule_note,year_start,year_end,action,notes
"""
from pathlib import Path
import csv
import re

DATA_DIR = Path(__file__).resolve().parents[1] / '2025_Problem_C_Data'
IN_PATH = DATA_DIR / 'country_mapping.csv'
OUT_PATH = DATA_DIR / 'country_mapping_cleaned.csv'


def load_candidates():
    candidates = set()
    mc = DATA_DIR / 'summerOly_medal_counts.csv'
    at = DATA_DIR / 'summerOly_athletes.csv'
    for fp, col in [(mc, 'NOC'), (at, 'Team')]:
        if fp.exists():
            with fp.open('r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    v = r.get(col)
                    if v:
                        candidates.add(v.strip())
    norm = {c: c for c in candidates}
    return norm


def is_probable_country(name):
    if not name or name.strip() == '':
        return False
    low = name.lower()
    non_country_tokens = ['club', 'boat', 'row', 'rowing', 'university', 'school', 'club', 'the', 'ii', 'iii', 'iv', 'v', 'team', 'polo', 'scull', 'cc', 'crew']
    if any(tok in low for tok in non_country_tokens):
        return False
    if re.search(r'\d', name):
        return False
    if ',' in name and len(name.split(',')) > 2:
        return False
    if re.match(r'^[A-Za-z &\-\(\)\.]+' , name):
        return True
    return False


def refine():
    candidates = load_candidates()
    rows = []
    if not IN_PATH.exists():
        print('No country_mapping.csv found at', IN_PATH)
        return
    with IN_PATH.open('r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for r in reader:
            orig = (r.get('original_name') or '').strip()
            suggested = (r.get('suggested_standard') or '').strip()
            note = (r.get('rule_note') or '').strip()
            action = 'needs_review'
            notes = ''

            if suggested in candidates or orig in candidates:
                action = 'keep'
            else:
                if not is_probable_country(orig):
                    action = 'remove'
                    notes = 'looks like club/boat or non-country'
                else:
                    matched = None
                    for c in candidates:
                        if c.lower() == orig.lower():
                            matched = c
                            break
                    if matched:
                        action = 'mapped'
                        suggested = matched
                        notes = 'case-insensitive match'
                    else:
                        action = 'needs_review'
                        notes = 'probable country but not in candidates'

            rows.append([orig, suggested, note, r.get('year_start',''), r.get('year_end',''), action, notes])

    with OUT_PATH.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['original_name','suggested_standard','rule_note','year_start','year_end','action','notes'])
        for row in rows:
            w.writerow(row)
    print('Wrote', OUT_PATH)


if __name__ == '__main__':
    refine()
