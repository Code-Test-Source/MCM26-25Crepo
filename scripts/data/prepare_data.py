#!/usr/bin/env python3
"""Prepare data: cleaning and feature engineering (moved into scripts/data).
This file is a copy of the original `scripts/prepare_data.py` implementation.
"""
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / '2025_Problem_C_Data'
OUT_DIR = ROOT / 'outputs'
OUT_DIR.mkdir(exist_ok=True)


def load_csv(name, **kwargs):
    path = DATA_DIR / name
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    last_exc = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last_exc = e
            continue
    raise last_exc


def clean_country(s):
    if pd.isna(s):
        return s
    return str(s).strip().replace('\u00A0', ' ')


def load_mapping():
    mapping_path = DATA_DIR / 'country_mapping.csv'
    if mapping_path.exists():
        try:
            m = load_csv('country_mapping.csv', dtype=str, engine='python').fillna('')
        except Exception:
            m = pd.read_csv(mapping_path, dtype=str, engine='python', on_bad_lines='skip').fillna('')
        lookup = {r['original_name'].strip(): (r.get('suggested_standard') or r['original_name'].strip()) for _, r in m.iterrows()}
        return lookup, m
    return {}, None


def build_panel(medal_counts_df, years, countries):
    idx = pd.MultiIndex.from_product([countries, years], names=['Country', 'Year'])
    panel = pd.DataFrame(index=idx).reset_index()
    df = medal_counts_df.rename(columns={'NOC': 'Country'})
    df = df[['Country', 'Year', 'Gold', 'Silver', 'Bronze', 'Total']]
    panel = panel.merge(df, on=['Country', 'Year'], how='left')
    panel[['Gold', 'Silver', 'Bronze', 'Total']] = panel[['Gold', 'Silver', 'Bronze', 'Total']].fillna(0).astype(int)
    return panel


def mark_hosts(panel, hosts_df):
    hosts_df = hosts_df.copy()
    hosts_df['HostCountry'] = hosts_df['Host'].astype(str).apply(lambda x: x.split(',')[-1].strip())
    host_map = dict(zip(hosts_df['Year'].astype(int), hosts_df['HostCountry']))
    panel['IsHost'] = 0
    for yr, country in host_map.items():
        if country:
            mask = (panel['Year'] == yr) & (panel['Country'].str.contains(country, na=False, regex=False))
            panel.loc[mask, 'IsHost'] = 1
    return panel


def compute_lags(panel):
    panel = panel.sort_values(['Country', 'Year']).reset_index(drop=True)
    for k in [1, 2, 3]:
        panel[f'Total_lag{k}'] = panel.groupby('Country')['Total'].shift(k).fillna(0).astype(int)
        panel[f'Gold_lag{k}'] = panel.groupby('Country')['Gold'].shift(k).fillna(0).astype(int)
    panel['Total_avg_3'] = panel.groupby('Country')['Total'].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
    panel['Total_std_3'] = panel.groupby('Country')['Total'].transform(lambda s: s.shift(1).rolling(3, min_periods=1).std()).fillna(0)
    panel['Best_historical'] = panel.groupby('Country')['Total'].transform(lambda s: s.shift(1).expanding().max().fillna(0))

    def slope(xs):
        ys = np.array(xs)
        if len(ys) < 2:
            return 0.0
        x = np.arange(len(ys))
        try:
            m = np.polyfit(x, ys, 1)[0]
        except Exception:
            m = 0.0
        return float(m)

    slopes = []
    for _, g in panel.groupby('Country'):
        totals = g['Total'].tolist()
        for i in range(len(totals)):
            hist = totals[max(0, i-3):i]
            slopes.append(slope(hist))
    panel['Total_trend_slope'] = slopes
    return panel


def load_programs_events(programs_df):
    years = [int(c) for c in programs_df.columns if c.isdigit()]
    te = programs_df[programs_df['Sport'].str.strip().str.lower() == 'total events']
    total_events = {}
    if not te.empty:
        row = te.iloc[0]
        for y in years:
            try:
                total_events[y] = int(row[str(y)])
            except Exception:
                try:
                    total_events[y] = int(float(row[str(y)]))
                except Exception:
                    total_events[y] = np.nan
    return years, total_events, programs_df


def compute_dominant_metrics(panel, athletes_df, programs_df, total_events):
    medals = athletes_df[athletes_df['Medal'].notna() & (athletes_df['Medal'] != 'No medal')].copy()
    medals['MedalCount'] = 1
    dominant_events = []
    dominant_ratio = []
    new_sports = []
    sport_first_year = athletes_df.groupby('Sport')['Year'].min().to_dict()

    for idx, row in panel.iterrows():
        country = row['Country']
        year = int(row['Year'])
        hist = medals[(medals['Team'] == country) & (medals['Year'] < year)]
        top5 = hist.groupby('Sport')['MedalCount'].sum().nlargest(5).index.tolist()
        if len(top5) == 0:
            dominant_events.append(0)
            dominant_ratio.append(0.0)
        else:
            ev_count = 0
            for sp in top5:
                matches = programs_df[programs_df['Sport'].str.lower().str.contains(str(sp).lower(), na=False)]
                for _, mrow in matches.iterrows():
                    try:
                        ev_count += int(mrow.get(str(year), 0) or 0)
                    except Exception:
                        try:
                            ev_count += int(float(mrow.get(str(year), 0) or 0))
                        except Exception:
                            pass
            dominant_events.append(ev_count)
            te = total_events.get(year, np.nan)
            dominant_ratio.append(ev_count / te if te and te > 0 else 0.0)
        ns = sum(1 for s, y0 in sport_first_year.items() if int(y0) == year)
        new_sports.append(ns)

    panel['DominantEvents'] = dominant_events
    panel['DominantRatio'] = dominant_ratio
    panel['New_Sports_Count'] = new_sports
    return panel


def main():
    medal_counts = load_csv('summerOly_medal_counts.csv')
    hosts = load_csv('summerOly_hosts.csv')
    athletes = load_csv('summerOly_athletes.csv')
    programs = load_csv('summerOly_programs.csv')

    for col in ['Team', 'Name', 'Sport', 'Event']:
        if col in athletes.columns:
            athletes[col] = athletes[col].astype(str).apply(clean_country)

    medal_counts['NOC'] = medal_counts['NOC'].astype(str).apply(clean_country)
    hosts['Host'] = hosts['Host'].astype(str).apply(clean_country)

    mapping, mapping_df = load_mapping()
    if mapping:
        athletes['Team'] = athletes['Team'].apply(lambda x: mapping.get(x.strip(), x.strip()))
        medal_counts['NOC'] = medal_counts['NOC'].apply(lambda x: mapping.get(x.strip(), x.strip()))

    years = sorted(medal_counts['Year'].dropna().astype(int).unique().tolist())
    countries = sorted(medal_counts['NOC'].dropna().unique().tolist())

    panel = build_panel(medal_counts, years, countries)
    panel = mark_hosts(panel, hosts)
    panel = compute_lags(panel)

    yrs, total_events, programs_df = load_programs_events(programs)
    panel = compute_dominant_metrics(panel, athletes, programs_df, total_events)

    proc_dir = OUT_DIR / 'processed_data'
    proc_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(proc_dir / 'processed_data.csv', index=False)
    if mapping_df is not None:
        mapping_df.to_csv(proc_dir / 'country_mapping_used.csv', index=False)

    feat_desc = [
        ('Year', 'Olympic year', 'source: medal_counts/hosts'),
        ('Country', 'Country name (standardized)', 'country_mapping.csv'),
        ('Gold/Silver/Bronze/Total', 'Medal counts', 'summerOly_medal_counts.csv'),
        ('IsHost', 'Host flag for that year (1/0)', 'summerOly_hosts.csv'),
        ('Gold_lag1/2/3', 'Gold count in 1/2/3 previous editions', 'computed from panel'),
        ('Total_avg_3', 'Mean Total over past 3 editions', 'computed window'),
        ('Total_trend_slope', 'Linear slope of Total in last up to 3 editions', 'np.polyfit'),
        ('DominantEvents', 'Count of events in this year matching country top-5 sports (historical)', 'programs & athletes'),
    ]
    pd.DataFrame(feat_desc, columns=['feature_name', 'description', 'notes']).to_csv(proc_dir / 'feature_description.csv', index=False)

    miss = panel.isna().sum().reset_index()
    miss.columns = ['feature', 'missing_count']
    miss.to_csv(proc_dir / 'missing_report.csv', index=False)

    print('Wrote processed_data.csv, feature_description.csv, missing_report.csv to outputs/')


if __name__ == '__main__':
    main()
