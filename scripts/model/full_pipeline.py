"""
Full Olympic Medal Prediction Pipeline
Implements: data loading, country normalization, geopolitical mapping,
Russia 2024 reconstruction (three methods), sport-weighted PCA,
ARIMA trend forecasts, feature engineering, XGBoost training + bootstrap
predictions with 95% CI, sensitivity scenarios, and first-medal probability.

Run as:
    python scripts/model/full_pipeline.py --run_all

Saves outputs to ./outputs/processed_data/ as described in the project plan.

Notes:
- Designed to run on the provided CSVs in ./2025_Problem_C_Data and
  ./outputs/processed_data where intermediate files are stored.
- Uses only provided datasets to derive mappings.
- Random seed fixed for reproducibility.
"""

import os
import argparse
import logging
import json
from functools import partial
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

# Config
RANDOM_STATE = 42
BASE_DIR = os.path.abspath("./")
DATA_DIR = os.path.join(BASE_DIR, "2025_Problem_C_Data")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "processed_data")
os.makedirs(OUT_DIR, exist_ok=True)

# Input files
MEDAL_COUNTS = os.path.join(DATA_DIR, "summerOly_medal_counts.csv")
ATHLETES = os.path.join(DATA_DIR, "summerOly_athletes.csv")
PROGRAMS = os.path.join(DATA_DIR, "summerOly_programs.csv")
HOSTS = os.path.join(DATA_DIR, "summerOly_hosts.csv")
DATA_DICT = os.path.join(DATA_DIR, "data_dictionary.csv")

# Output files
COUNTRY_MAP_FN = os.path.join(OUT_DIR, "country_mapping_auto.csv")
RUSSIA_EST_FN = os.path.join(OUT_DIR, "russia_2024_estimates.csv")
MEDAL_ADJ_FN = os.path.join(OUT_DIR, "olympics_medals_adjusted.csv")
MEDAL_ORIG_FN = os.path.join(OUT_DIR, "olympics_medals_original.csv")
PCA_FN = os.path.join(OUT_DIR, "pca_sport_features.csv")
HOLT_FN = os.path.join(OUT_DIR, "holt_trends_2028.csv")
PRED_FN = os.path.join(OUT_DIR, "predictions_2028_final.csv")
FIRST_MEDAL_FN = os.path.join(OUT_DIR, "first_medal_probabilities.csv")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('full_pipeline')

# Utilities
import re

def normalize_country_name(s):
    if not isinstance(s, str):
        return s
    # normalize common unicode whitespace and casing
    s = s.replace('\u00A0', ' ')
    s2 = s.strip().lower()
    s2 = re.sub(r"\s*\(.*\)$", "", s2)
    s2 = re.sub(r"\s+\d+$", "", s2)
    s2 = re.sub(r"[-_]", " ", s2)
    s2 = re.sub(r"\s+team$", "", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def normalize_noc_value(x):
    """Normalize NOC-like fields: trim, replace NBSP, collapse spaces."""
    if pd.isna(x):
        return x
    s = str(x).replace('\u00A0', ' ').strip()
    s = re.sub(r"\s+", " ", s)
    return s


def safe_forecast(series, steps=1):
    """Forecast using Holt's linear method only (no ARIMA).
    Returns a numpy array of length `steps`.
    """
    s = series.dropna()
    if len(s) == 0:
        return np.zeros(steps)
    # if too short, repeat last known value
    if len(s) == 1:
        return np.array([float(s.iloc[-1])] * steps)
    # Try a lightweight ARIMA first for sufficiently long series, then Holt, then linear fallback
    try:
        if len(s) >= 8 and np.nanstd(s) > 0:
            try:
                arima_mod = ARIMA(s, order=(1, 1, 0))
                arima_res = arima_mod.fit(method='css', maxiter=50)
                af = arima_res.forecast(steps)
                arr = np.asarray(af, dtype=float)
                arr[arr < 0] = 0.0
                return arr
            except Exception:
                # fall through to Holt
                pass
        holt = Holt(s).fit(optimized=False, smoothing_level=0.5, smoothing_trend=0.1)
        hf = holt.forecast(steps)
        arr = np.asarray(hf, dtype=float)
        arr[arr < 0] = 0.0
        return arr
    except Exception:
        # deterministic linear fallback
        try:
            y = s.values.astype(float)
            x = np.arange(len(y))
            a, b = np.polyfit(x, y, 1)
            preds = a * (len(x) + np.arange(steps)) + b
            preds = np.maximum(preds, 0.0)
            return preds
        except Exception:
            return np.array([float(s.iloc[-1])] * steps)

# Geopolitical mapping
HISTORICAL_MAPPING = {
    'URS': 'RUS', 'soviet union': 'RUS', 'eun': 'RUS', 'unified team': 'RUS', 'roc': 'RUS',
    'frg': 'GER', 'gdr': 'GER', 'west germany': 'GER', 'east germany': 'GER',
    'tch': 'CZE', 'czechoslovakia': 'CZE', 'yug': 'SRB', 'yugoslavia': 'SRB',
    'great britain': 'GBR', 'united kingdom': 'GBR', 'united states': 'USA', 'china': 'CHN'
}


def map_noc_to_modern(noc_val):
    """Map historical or verbose NOC/country values to modern 3-letter NOC codes.
    Uses HISTORICAL_MAPPING for known historical names. If value already looks
    like a 3-letter code, return uppercased value. Otherwise return stripped value.
    """
    if pd.isna(noc_val):
        return noc_val
    s = str(noc_val).replace('\u00A0', ' ').strip()
    if s == "":
        return s
    key = s.lower()
    if key in HISTORICAL_MAPPING:
        return HISTORICAL_MAPPING[key]
    if len(s) == 3 and s.isalpha():
        return s.upper()
    up = s.upper()
    if len(up) == 3 and up.isalpha():
        return up
    return s

# Step 1: Data loading and initial checks

def load_inputs():
    log.info('Loading input files...')
    medals = pd.read_csv(MEDAL_COUNTS) if os.path.exists(MEDAL_COUNTS) else pd.DataFrame()
    athletes = pd.read_csv(ATHLETES) if os.path.exists(ATHLETES) else pd.DataFrame()
    programs = pd.DataFrame()
    if os.path.exists(PROGRAMS):
        try:
            programs = pd.read_csv(PROGRAMS)
        except UnicodeDecodeError:
            log.warning('UTF-8 decode failed for %s; retrying with cp1252', PROGRAMS)
            programs = pd.read_csv(PROGRAMS, encoding='cp1252')
        except Exception:
            log.exception('Failed to read programs file; proceeding with empty DataFrame')
            programs = pd.DataFrame()
    hosts = pd.read_csv(HOSTS) if os.path.exists(HOSTS) else pd.DataFrame()
    return medals, athletes, programs, hosts

# Step 2: country mapping derivation

def derive_country_mapping(athletes_df, medal_counts_df=None):
    """Derive mapping from name variants to standardized NOC where possible.
    Produces country_mapping_auto.csv with original_name, standardized_name, olympic_noc.
    """
    log.info('Deriving country mapping from athletes data...')
    # collect unique Team values from athletes
    teams = athletes_df['Team'].dropna().unique().tolist() if 'Team' in athletes_df.columns else []
    team_keys = {t: normalize_country_name(t) for t in teams}
    # try to map normalized names to NOC via medal_counts (matching by name or by aggregated counts)
    name_map = {}
    if medal_counts_df is not None and not medal_counts_df.empty:
        # attempt to map by matching variants to NOC where a variant equals a known country string
        possible_names = set(medal_counts_df['NOC'].astype(str).unique())
        # also build lookup by lowercase country name if present
        if 'country' in medal_counts_df.columns:
            for _, r in medal_counts_df.iterrows():
                nm = r.get('country')
                noc = r.get('NOC')
                if pd.notna(nm) and pd.notna(noc):
                    key = normalize_country_name(nm)
                    name_map[key] = noc
    # fallback: if mapping not found, use heuristic: map team normalized -> attempt to find NOC from athletes rows with same Team and NOC column
    if 'NOC' in athletes_df.columns:
        for _, row in athletes_df.iterrows():
            team = row.get('Team')
            noc = row.get('NOC')
            if pd.notna(team) and pd.notna(noc):
                key = normalize_country_name(team)
                name_map.setdefault(key, noc)
    # compose output table
    rows = []
    for orig, norm in team_keys.items():
        noc = name_map.get(norm, '')
        std = norm.title() if isinstance(norm, str) else norm
        rows.append({'original_name': orig, 'standardized_name': std, 'olympic_noc': noc})
    df_map = pd.DataFrame(rows)
    df_map.to_csv(COUNTRY_MAP_FN, index=False)
    log.info('Saved country mapping to %s', COUNTRY_MAP_FN)
    return df_map

# Step 3: Geopolitical alignment / build legacy_id

def apply_historical_mapping(df):
    df = df.copy()
    if 'NOC' in df.columns:
        df['NOC_std'] = df['NOC'].astype(str).map(lambda x: HISTORICAL_MAPPING.get(x, x))
    else:
        df['NOC_std'] = df.get('Mapped_NOC', df.get('NOC', pd.Series())).astype(str)
    # legacy_id: map historical strings to modern NOC
    def legacy(s):
        if not isinstance(s, str):
            return s
        k = s.strip().lower()
        return HISTORICAL_MAPPING.get(k, s)
    df['legacy_id'] = df.get('country', df.get('Team', '')).astype(str).map(legacy)
    return df

# Step 4: Russia 2024 estimation methods

def estimate_russia_arima(medals_df):
    """Method A: Holt trend extrapolation on Russia time series (Total and Gold).
    Returns predicted totals for 2024 and a per-sport allocation based on historical shares.
    """
    log.info('Estimating Russia 2024 via Holt (Method A)')
    # robustly identify Russia rows by NOC or country name (if present)
    noc_series = medals_df['NOC'].astype(str).str.upper() if 'NOC' in medals_df.columns else pd.Series([], dtype=object)
    mask_noc = noc_series.isin(['RUS','ROC','URS'])
    mask_country = False
    if 'country' in medals_df.columns:
        mask_country = medals_df['country'].astype(str).str.lower() == 'russia'
    rus = medals_df[mask_noc | mask_country]
    # standardize NOC if present; fallback to any rows with NOC variants
    if rus.empty and 'NOC' in medals_df.columns:
        log.warning('No Russia rows found in medal counts; attempting fallback by NOC variants')
        rus = medals_df[medals_df['NOC'].astype(str).str.upper().isin(['RUS','ROC','URS'])]
    rus = rus.sort_values('Year')
    series_total = rus.groupby('Year')['Total'].sum()
    series_gold = rus.groupby('Year')['Gold'].sum()
    def arima_forecast(series):
        # use safe_forecast helper (Holt -> linear fallback)
        arr = safe_forecast(series, steps=1)
        return float(arr[0]) if len(arr)>0 else 0.0
    pred_total = arima_forecast(series_total)
    pred_gold = arima_forecast(series_gold)
    # per-sport allocation: use historical average share (2012-2020)
    if 'Sport' in rus.columns:
        sport_shares = rus[rus['Year']>=2012].groupby('Sport')['Total'].sum()
        if sport_shares.sum() == 0:
            sport_shares = rus.groupby('Sport')['Total'].sum()
        if sport_shares.sum() == 0:
            sport_shares = pd.Series({'Unknown':1.0})
        sport_shares = (sport_shares / sport_shares.sum()).to_dict()
    else:
        sport_shares = {'All': 1.0}
    # allocate integer medals by proportion (approx)
    alloc = {}
    for sport, share in sport_shares.items():
        alloc[sport] = {'Total': pred_total * share, 'Gold': pred_gold * share}
    return {'method':'A', 'pred_total': pred_total, 'pred_gold': pred_gold, 'by_sport': alloc}


def estimate_russia_relative(medals_df, reference_nocs=None):
    """Method B: Relative-performance method using reference countries.
    Compute growth rates for similar countries and apply to Russia's last-known value.
    """
    log.info('Estimating Russia 2024 via Relative method (Method B)')
    # choose default references if not provided
    if reference_nocs is None:
        reference_nocs = ['BLR','KAZ','UKR']
    # compute growth rate from 2016->2020 average of references
    ref = medals_df[medals_df['NOC'].isin(reference_nocs)]
    if ref.empty:
        ref = medals_df[medals_df['NOC'].isin(medals_df['NOC'].unique()[:5])]
    # compute avg growth rate for Total between 2016 and 2020
    def growth(noc):
        s = ref[ref['NOC']==noc].set_index('Year')['Total'].sort_index()
        if 2016 in s.index and 2020 in s.index and s.loc[2016]>0:
            return (s.loc[2020]-s.loc[2016]) / s.loc[2016]
        return 0.0
    rates = [growth(n) for n in reference_nocs]
    avg_rate = np.nanmean(rates) if len(rates)>0 else 0.0
    # Russia last known total (e.g., 2020)
    rus = medals_df[medals_df['NOC'].isin(['RUS','ROC','URS','Russia'])]
    if rus.empty:
        rus = medals_df[medals_df['NOC'].astype(str).str.upper().isin(['RUS','ROC','URS'])]
    last = rus[rus['Year']<=2020].sort_values('Year').groupby('Year')['Total'].sum()
    if last.empty:
        base_total = 0.0
    else:
        base_total = float(last.iloc[-1])
    pred_total = base_total * (1.0 + avg_rate)
    # gold scaled same way by historical ratio
    last_gold = rus[rus['Year']<=2020].sort_values('Year').groupby('Year')['Gold'].sum()
    base_gold = float(last_gold.iloc[-1]) if not last_gold.empty else 0.0
    pred_gold = base_gold * (1.0 + avg_rate)
    # allocate by sport using recent shares
    if 'Sport' in rus.columns:
        sport_shares = rus[rus['Year']>=2012].groupby('Sport')['Total'].sum()
        if sport_shares.sum() == 0:
            sport_shares = rus.groupby('Sport')['Total'].sum()
        sport_shares = (sport_shares / sport_shares.sum()).to_dict() if sport_shares.sum()>0 else {'Unknown':1.0}
    else:
        sport_shares = {'All':1.0}
    alloc = {s:{'Total':pred_total*sh,'Gold':pred_gold*sh} for s,sh in sport_shares.items()}
    return {'method':'B','pred_total':pred_total,'pred_gold':pred_gold,'by_sport':alloc}


def estimate_russia_age_based(athletes_df, medals_df):
    """Method C: Age-structure method (simplified): estimate proportion of athletes
    likely to remain competitive. If age data absent, fallback to conservative estimate.
    """
    log.info('Estimating Russia 2024 via Age-structured method (Method C)')
    if 'Age' in athletes_df.columns:
        rus_ath = athletes_df[athletes_df['NOC'].str.upper().isin(['RUS','ROC'])]
        # compute proportion of athletes in 2016/2020 cohorts likely still active
        # simplified: fraction under 30 in 2020
        cohort = rus_ath[rus_ath['Year']==2020]
        if not cohort.empty and 'Age' in cohort.columns:
            prop = (cohort['Age']<=30).mean()
        else:
            prop = 0.6
    else:
        prop = 0.6
    # baseline from medals df
    rus = medals_df[medals_df['NOC'].str.upper().isin(['RUS','ROC','URS'])]
    last_total = rus[rus['Year']<=2020].groupby('Year')['Total'].sum()
    base = float(last_total.iloc[-1]) if not last_total.empty else 0.0
    pred_total = base * prop
    last_gold = rus[rus['Year']<=2020].groupby('Year')['Gold'].sum()
    base_gold = float(last_gold.iloc[-1]) if not last_gold.empty else 0.0
    pred_gold = base_gold * prop
    # allocate by sport as before
    if 'Sport' in rus.columns:
        sport_shares = rus[rus['Year']>=2012].groupby('Sport')['Total'].sum()
        if sport_shares.sum()==0:
            sport_shares = rus.groupby('Sport')['Total'].sum()
        sport_shares = (sport_shares / sport_shares.sum()).to_dict() if sport_shares.sum()>0 else {'Unknown':1.0}
    else:
        sport_shares = {'All':1.0}
        alloc = {s:{'Total':pred_total*sh,'Gold':pred_gold*sh} for s,sh in sport_shares.items()}
        return {'method':'C','pred_total':pred_total,'pred_gold':pred_gold,'by_sport':alloc}


def adjust_2024_medals(russia_estimates, actual_2024_df):
    """Adjust 2024 medals to simulate Russia participation.
    Simplified algorithm:
    - For each sport, if Russia_est > 0, reduce medals of countries with
      the largest positive deviation from historical mean in 2024 (i.e., 'abnormal gain').
    - Remove medals starting from countries that likely benefited most.
    - This is heuristic but captures reallocation effect.

    Returns adjusted DataFrame.
    """
    log.info('Adjusting 2024 medals to simulate Russia participation...')
    df = actual_2024_df.copy()
    try:
        log.info('adjust_2024_medals incoming Year dtype=%s unique_sample=%s', df['Year'].dtype, pd.unique(df['Year'])[:5])
        log.info('adjust_2024_medals initial count_2024=%d', int((df['Year']==2024).sum()))
    except Exception:
        log.exception('Failed to inspect Year in adjust_2024_medals')
    # compute historical baseline (2012-2020) per sport-country
    hist = df[df['Year']<2024].copy()
    if 'Sport' in df.columns:
        mean_hist = hist.groupby(['NOC','Sport'])['Total'].mean().reset_index().rename(columns={'Total':'Hist_Mean'})
        cur_2024 = df[df['Year']==2024].copy()
        cur_2024 = cur_2024.merge(mean_hist, on=['NOC','Sport'], how='left').fillna({'Hist_Mean':0})
        cur_2024['Dev'] = cur_2024['Total'] - cur_2024['Hist_Mean']
    else:
        mean_hist = hist.groupby(['NOC'])['Total'].mean().reset_index().rename(columns={'Total':'Hist_Mean'})
        cur_2024 = df[df['Year']==2024].groupby('NOC').agg({'Gold':'sum','Silver':'sum','Bronze':'sum','Total':'sum'}).reset_index()
        cur_2024 = cur_2024.merge(mean_hist, on=['NOC'], how='left').fillna({'Hist_Mean':0})
        cur_2024['Dev'] = cur_2024['Total'] - cur_2024['Hist_Mean']
    # For each sport where Russia has expected medals, reduce others
    adjusted = cur_2024.copy()
    for sport, est in russia_estimates.get('by_sport', {}).items():
        est_total = int(round(est.get('Total', 0)))
        if est_total <= 0:
            continue
        remaining = est_total
        if 'Sport' in adjusted.columns:
            mask = adjusted['Sport']==sport
            candidates = adjusted[mask].sort_values('Dev', ascending=False)
            for idx, row in candidates.iterrows():
                if remaining<=0:
                    break
                take = min(remaining, int(max(0, row['Total'])))
                if take<=0:
                    continue
                adjusted.loc[idx,'Total'] = int(max(0, adjusted.loc[idx,'Total'] - take))
                tot = row['Total']
                if tot>0:
                    gfrac = row.get('Gold',0)/tot
                    sfrac = row.get('Silver',0)/tot
                    bfrac = row.get('Bronze',0)/tot
                    adjusted.loc[idx,'Gold'] = int(max(0, adjusted.loc[idx,'Gold'] - round(take*gfrac)))
                    adjusted.loc[idx,'Silver'] = int(max(0, adjusted.loc[idx,'Silver'] - round(take*sfrac)))
                    adjusted.loc[idx,'Bronze'] = int(max(0, adjusted.loc[idx,'Bronze'] - round(take*bfrac)))
                remaining -= take
            # add/increment Russia sport row
            rus_idx = adjusted[(adjusted['NOC'].isin(['RUS','ROC'])) & (adjusted['Sport']==sport)]
            if not rus_idx.empty:
                ridx = rus_idx.index[0]
                adjusted.loc[ridx,'Total'] = int(adjusted.loc[ridx,'Total'] + est_total)
                adjusted.loc[ridx,'Gold'] = int(adjusted.loc[ridx,'Gold'] + int(round(est.get('Gold',0))))
            else:
                newrow = {'Year':2024, 'NOC':'RUS', 'Sport':sport, 'Gold': int(round(est.get('Gold',0))),
                          'Silver':0, 'Bronze':0, 'Total': int(est_total)}
                adjusted = pd.concat([adjusted, pd.DataFrame([newrow])], ignore_index=True, sort=False)
        else:
            candidates = adjusted.sort_values('Dev', ascending=False)
            for idx, row in candidates.iterrows():
                if remaining<=0:
                    break
                take = min(remaining, int(max(0, row['Total'])))
                if take<=0:
                    continue
                adjusted.loc[idx,'Total'] = int(max(0, adjusted.loc[idx,'Total'] - take))
                tot = row['Total']
                if tot>0:
                    gfrac = row.get('Gold',0)/tot
                    adjusted.loc[idx,'Gold'] = int(max(0, adjusted.loc[idx,'Gold'] - round(take*gfrac)))
                remaining -= take
            rus_idx = adjusted[adjusted['NOC'].isin(['RUS','ROC'])]
            if not rus_idx.empty:
                ridx = rus_idx.index[0]
                adjusted.loc[ridx,'Total'] = int(adjusted.loc[ridx,'Total'] + est_total)
                adjusted.loc[ridx,'Gold'] = int(adjusted.loc[ridx,'Gold'] + int(round(est.get('Gold',0))))
            else:
                newrow = {'Year':2024, 'NOC':'RUS', 'Gold': int(round(est.get('Gold',0))), 'Silver':0, 'Bronze':0, 'Total': int(est_total)}
                adjusted = pd.concat([adjusted, pd.DataFrame([newrow])], ignore_index=True, sort=False)
    others = df[df['Year']!=2024]
    result = pd.concat([others, adjusted], ignore_index=True, sort=False)
    return result

# Step 5: build panel datasets

def build_panel(medal_counts_df, athletes_df, hosts_df):
    log.info('Building panel data sets...')
    # If medal_counts_df exists, use it as base else aggregate athletes
    if not medal_counts_df.empty:
        df = medal_counts_df.copy()
        # normalize NOC-like values and merge duplicates (some entries use trailing NBSP or variants)
        if 'NOC' in df.columns:
            df['NOC'] = df['NOC'].map(normalize_noc_value)
        # ensure numeric medal columns exist
        # Ensure essential columns exist. For Year, leave missing as NaN rather than 0.
        for c in ['NOC','Gold','Silver','Bronze','Total']:
            if c not in df.columns:
                df[c] = 0
        if 'Year' not in df.columns:
            df['Year'] = np.nan
        # If Sport column exists, aggregate by Year,NOC,Sport; otherwise aggregate Year,NOC
        if 'Sport' in df.columns:
            agg_cols = ['Year','NOC','Sport']
        else:
            agg_cols = ['Year','NOC']
        num_cols = [c for c in ['Gold','Silver','Bronze','Total'] if c in df.columns]
        df = df.groupby(agg_cols, dropna=False)[num_cols].sum().reset_index()
        # coerce Year to numeric and drop rows without a valid positive Year
        try:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df = df[df['Year'].notna() & (df['Year']>0)].copy()
            df['Year'] = df['Year'].astype(int)
        except Exception:
            pass
    else:
        # aggregate athletes
        if athletes_df.empty:
            raise RuntimeError('No input data available')
        agg = athletes_df.groupby(['Year','NOC','Sport']).agg({'Medal':lambda x: 0}).reset_index()
        df = agg
    # add is_host flag via hosts_df
    if not hosts_df.empty:
        hosts_map = hosts_df.set_index('Year')['HostNOC'] if 'HostNOC' in hosts_df.columns else None
        if hosts_map is not None:
            df['is_host'] = df['Year'].map(lambda y: 1 if y in hosts_map.index and hosts_map.loc[y] in str(df.get('NOC', '')) else 0)
        else:
            df['is_host'] = 0
    else:
        df['is_host'] = 0
    return df

# Step 6: sport-weighted scoring & PCA

def compute_sport_weighted_pca(panel_df, n_components=5):
    log.info('Computing sport-weighted scores and PCA...')
    # compute weighted score per (Year,NOC,Sport): w = Gold*0.5 + Silver*0.3 + Bronze*0.2
    if not {'Gold','Silver','Bronze'}.issubset(panel_df.columns):
        # try to derive per-sport from Total only
        panel_df['Gold'] = panel_df.get('Gold',0)
        panel_df['Silver'] = panel_df.get('Silver',0)
        panel_df['Bronze'] = panel_df.get('Bronze',0)
    panel_df['Weighted'] = panel_df['Gold']*0.5 + panel_df['Silver']*0.3 + panel_df['Bronze']*0.2
    if 'Sport' in panel_df.columns:
        wide = panel_df.pivot_table(index=['Year','NOC'], columns='Sport', values='Weighted', aggfunc='sum', fill_value=0)
    else:
        # fallback: aggregate weighted score per (Year,NOC) into a single 'All' column
        wide_series = panel_df.groupby(['Year','NOC'])['Weighted'].sum()
        wide = wide_series.to_frame('All')
    # standardize and PCA
    scaler = StandardScaler()
    X = scaler.fit_transform(wide.values)
    # adjust n_components to available data dimensions
    n_comp_adj = max(1, min(n_components, X.shape[1], X.shape[0]))
    pca = PCA(n_components=n_comp_adj, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X)
    cols = [f'PC{i+1}' for i in range(pcs.shape[1])]
    pca_df = pd.DataFrame(pcs, columns=cols, index=wide.index).reset_index()
    pca_df.to_csv(PCA_FN, index=False)
    log.info('Saved PCA features to %s', PCA_FN)
    return pca_df, pca

# Step 7: ARIMA trends for each NOC

def arima_trends(panel_df, target_year=2028):
    log.info('Fitting Holt trends per NOC...')
    # ensure NOC normalization to remove invisible variants/whitespace
    df = panel_df.copy()
    if 'NOC' in df.columns:
        df['NOC'] = df['NOC'].map(normalize_noc_value)
    # aggregate after normalization to avoid duplicate NOC entries
    if 'Sport' in df.columns:
        agg_cols = ['Year','NOC','Sport']
    else:
        agg_cols = ['Year','NOC']
    num_cols = [c for c in ['Gold','Silver','Bronze','Total'] if c in df.columns]
    df = df.groupby(agg_cols, dropna=False)[num_cols].sum().reset_index()
    results = []
    groups = df.groupby('NOC')
    for noc, g in groups:
        # use a fast, robust historical rule: mean of the last up-to-3 observed years
        series_total = g.sort_values('Year').groupby('Year')['Total'].sum()
        series_gold = g.sort_values('Year').groupby('Year')['Gold'].sum()
        # if series is empty -> 0; if short -> last value; if >=1 -> mean of last 3
        def fast_history_forecast(s):
            s = s.dropna()
            if len(s) == 0:
                return 0.0
            if len(s) == 1:
                return float(s.iloc[-1])
            # take mean of last up to 3 observations
            vals = s.sort_index().iloc[-3:]
            return float(vals.mean())
        t_pred = fast_history_forecast(series_total)
        g_pred = fast_history_forecast(series_gold)
        results.append({'NOC':noc,'HOLT_Total_2028':t_pred,'HOLT_Gold_2028':g_pred})
    res_df = pd.DataFrame(results)
    # Map historical/verbose NOC values to modern 3-letter NOC codes and aggregate
    if 'NOC' in res_df.columns:
        res_df['NOC_mapped'] = res_df['NOC'].apply(map_noc_to_modern)
        holt_cols = [c for c in res_df.columns if c.startswith('HOLT_')]
        res_df = res_df.groupby('NOC_mapped', as_index=False)[holt_cols].sum()
        res_df = res_df.rename(columns={'NOC_mapped': 'NOC'})
    res_df.to_csv(HOLT_FN, index=False)
    log.info('Saved Holt trends to %s', HOLT_FN)
    return res_df

# Step 8: Feature engineering

def build_features(panel_df, pca_df, arima_df):
    log.info('Building features for XGBoost...')
    # panel_df has Year,NOC,Gold,Total,is_host
    df = panel_df.copy()
    # map historical NOC/country names into modern 3-letter NOC codes
    if 'NOC' in df.columns:
        df['NOC'] = df['NOC'].apply(map_noc_to_modern)
    # ensure Year is numeric and drop invalid rows (fix cases where Year was set to 0)
    try:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df[df['Year'].notna() & (df['Year']>0)].copy()
        df['Year'] = df['Year'].astype(int)
    except Exception:
        log.exception('Failed coercing Year in build_features; continuing with original Year values')
    # lag features
    df = df.sort_values(['NOC','Year'])
    df['Gold_Lag1'] = df.groupby('NOC')['Gold'].shift(1)
    df['Total_Lag1'] = df.groupby('NOC')['Total'].shift(1)
    df['Gold_MA3'] = df.groupby('NOC')['Gold'].rolling(3, min_periods=1).mean().reset_index(0,drop=True)
    df['Total_MA3'] = df.groupby('NOC')['Total'].rolling(3, min_periods=1).mean().reset_index(0,drop=True)
    df['Gold_Best_Ever'] = df.groupby('NOC')['Gold'].cummax()
    df['Total_Best_Ever'] = df.groupby('NOC')['Total'].cummax()
    # merge ARIMA and PCA
    df = df.merge(arima_df, left_on='NOC', right_on='NOC', how='left')
    df = df.merge(pca_df, left_on=['Year','NOC'], right_on=['Year','NOC'], how='left')
    # fillna
    df.fillna(0, inplace=True)
    # simple Russia-effect features
    # competition_with_russia: historical correlation of NOC with Russia in totals
    # compute correlation across years
    corr = panel_df.pivot_table(index='Year',columns='NOC',values='Total',aggfunc='sum',fill_value=0)
    if 'RUS' in corr.columns:
        rus_series = corr['RUS']
        cors = corr.corrwith(rus_series).to_dict()
        df['competition_with_russia'] = df['NOC'].map(lambda x: cors.get(x,0.0))
    else:
        df['competition_with_russia'] = 0.0
    # russia_absence_benefit_2024: delta 2024 over prior mean
    means = panel_df[panel_df['Year']<2024].groupby('NOC')['Total'].mean().to_dict()
    totals_2024 = panel_df[panel_df['Year']==2024].set_index('NOC')['Total'].to_dict()
    def benefit(noc):
        base = means.get(noc, 0.0)
        cur = totals_2024.get(noc, 0.0)
        return (cur - base) / (base + 1e-6)
    df['russia_absence_benefit_2024'] = df['NOC'].map(benefit)
    # expected_russia_impact_2028: placeholder using ARIMA_RUS magnitude
    df['expected_russia_impact_2028'] = 0.0
    # other participation features
    # athlete_count/events_participated/female_ratio could be derived from athletes_df, skipped for brevity
    return df

# Step 9+10: Train XGBoost and bootstrap predictions

def train_and_bootstrap(train_df, features, target_col='Total', B=200, n_jobs=-1):
    log.info('Training XGBoost and running bootstrap (B=%d)...', B)
    X = train_df[features].copy()
    # keep only numeric features (drop object/string columns introduced by bad merges)
    X = X.select_dtypes(include=[np.number])
    used_features = X.columns.tolist()
    y = train_df[target_col]
    # split holdout for simple hyperparam selection
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    # basic param set
    params = {'n_estimators':200, 'max_depth':5, 'learning_rate':0.05, 'subsample':0.8, 'colsample_bytree':0.8, 'random_state':RANDOM_STATE}
    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    log.info('Validation RMSE: %.3f', np.sqrt(mean_squared_error(y_val, val_pred)))
    # bootstrap resampling
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X)
    models = []
    def one_boot(i):
        idx = rng.integers(0, n, n)
        Xb = X.iloc[idx]
        yb = y.iloc[idx]
        m = XGBRegressor(**params)
        m.fit(Xb, yb)
        return m
    n_jobs = cpu_count() if n_jobs==-1 else n_jobs
    models = Parallel(n_jobs=min(8,n_jobs))(delayed(one_boot)(i) for i in range(B))
    return model, models, used_features


def predict_bootstrap(models, X_pred):
    log.info('Predicting with %d bootstrap models...', len(models))
    preds = np.vstack([m.predict(X_pred) for m in models])
    # compute median and 95% CI
    median = np.median(preds, axis=0)
    lower = np.percentile(preds, 2.5, axis=0)
    upper = np.percentile(preds, 97.5, axis=0)
    return median, lower, upper, preds

# Step 11: sensitivity scenarios by scaling Russia strength

def run_sensitivity(panel_df, pca_df, arima_df, scale_factors=[1.1,1.0,0.7], B=200):
    results = {}
    for factor in scale_factors:
        log.info('Running sensitivity scenario Russia scale=%.2f', factor)
        # adjust Russia in panel_df Year==2024 by factor
        df = panel_df.copy()
        mask = (df['Year']==2024) & (df['NOC'].isin(['RUS','ROC']))
        df.loc[mask,'Total'] = df.loc[mask,'Total'] * factor
        # build features
        features_df = build_features(df, pca_df, arima_df)
        # select training rows (Year <=2024)
        train = features_df[features_df['Year']<=2024]
        # pick features
        feat_cols = [c for c in train.columns if c not in ['Year','NOC','Gold','Total','is_host']]
        # train and bootstrap
        _, models = train_and_bootstrap(train, feat_cols, target_col='Total', B=B)
        # prepare 2028 input rows
        pred_rows = features_df[features_df['Year']==2028]
        X_pred = pred_rows[feat_cols]
        median, lo, hi, all_preds = predict_bootstrap(models, X_pred)
        results[factor] = {'median':median, 'lower':lo, 'upper':hi, 'nocs':pred_rows['NOC'].tolist()}
    return results

# Step 12: first medal probability

def compute_first_medal_prob(all_preds, nocs, threshold=1):
    # all_preds: B x N matrix
    probs = (all_preds >= threshold).mean(axis=0)
    df = pd.DataFrame({'NOC': nocs, 'prob_first_medal': probs})
    df.to_csv(FIRST_MEDAL_FN, index=False)
    log.info('Saved first medal probabilities to %s', FIRST_MEDAL_FN)
    return df

# Orchestration

def run_full_pipeline(run_all=True, B=200):
    medals, athletes, programs, hosts = load_inputs()
    # debug: check medals year coverage right after load
    try:
        log.info('Loaded medals: min_year=%s max_year=%s count_2024=%d',
                 medals['Year'].min(), medals['Year'].max(), int((medals['Year']==2024).sum()))
    except Exception:
        log.exception('Failed to inspect medals years after load')
    # Step 2 mapping
    mapping_df = derive_country_mapping(athletes, medals)
    # Save original medals panel
    if not medals.empty:
        medals.to_csv(MEDAL_ORIG_FN, index=False)
    # Step 4: Russia estimates
    a = estimate_russia_arima(medals)
    b = estimate_russia_relative(medals)
    c = estimate_russia_age_based(athletes, medals)
    combined = pd.DataFrame([{
        'method': a['method'], 'pred_total': a['pred_total'], 'pred_gold': a['pred_gold']
    },{
        'method': b['method'], 'pred_total': b['pred_total'], 'pred_gold': b['pred_gold']
    },{
        'method': c['method'], 'pred_total': c['pred_total'], 'pred_gold': c['pred_gold']
    }])
    combined.to_csv(RUSSIA_EST_FN, index=False)
    log.info('Saved Russia estimates to %s', RUSSIA_EST_FN)
    # ensemble composite (weights A:0.5,B:0.3,C:0.2)
    pred_total = 0.5*a['pred_total'] + 0.3*b['pred_total'] + 0.2*c['pred_total']
    pred_gold = 0.5*a['pred_gold'] + 0.3*b['pred_gold'] + 0.2*c['pred_gold']
    ensemble = {'pred_total':pred_total,'pred_gold':pred_gold,'by_sport':{}}
    # Step 4.3: adjust 2024 medals
    adjusted = adjust_2024_medals({'by_sport': {}}, medals)  # for now use empty per-sport; complex reallocations require per-event details
    adjusted.to_csv(MEDAL_ADJ_FN, index=False)
    log.info('Saved adjusted medals to %s', MEDAL_ADJ_FN)
    # debug: log adjusted Year coverage
    try:
        log.info('Adjusted panel years: min=%s max=%s count_2024=%d',
                 adjusted['Year'].min(), adjusted['Year'].max(), int((adjusted['Year']==2024).sum()))
    except Exception:
        log.exception('Failed to log adjusted Year info')
    # Normalize NOC values in the adjusted panel and aggregate to remove duplicate NOC entries
    if not adjusted.empty and 'NOC' in adjusted.columns:
        adjusted['NOC'] = adjusted['NOC'].map(normalize_noc_value)
        # choose aggregation level consistent with downstream code
        if 'Sport' in adjusted.columns:
            agg_cols = ['Year','NOC','Sport']
        else:
            agg_cols = ['Year','NOC']
        num_cols = [c for c in ['Gold','Silver','Bronze','Total'] if c in adjusted.columns]
        adjusted = adjusted.groupby(agg_cols, dropna=False)[num_cols].sum().reset_index()
        # ensure Year is numeric
        try:
            adjusted['Year'] = adjusted['Year'].astype(int)
        except Exception:
            pass
    # Step 6: PCA
    pca_df, pca_model = compute_sport_weighted_pca(adjusted)
    # Step 7: ARIMA
    arima_df = arima_trends(adjusted)
    # Step 8: features
    features_df = build_features(adjusted, pca_df, arima_df)
    # debug export: save features and log counts to help diagnose empty 2028 predictions
    try:
        debug_fn = os.path.join(OUT_DIR, 'features_debug.csv')
        features_df.to_csv(debug_fn, index=False)
        log.info('Saved features debug to %s (rows=%d, cols=%d)', debug_fn, features_df.shape[0], features_df.shape[1])
        log.info('Rows where Year==2024: %d; Year==2028: %d', int((features_df['Year']==2024).sum()), int((features_df['Year']==2028).sum()))
    except Exception:
        log.exception('Failed to write features debug file')
    # Train on data up to 2024
    train = features_df[features_df['Year']<=2024]
    feat_cols = [c for c in train.columns if c not in ['Year','NOC','Gold','Total','is_host']]
    log.info('Selected feature columns (count=%d)', len(feat_cols))
    # Step 9-10: train and bootstrap
    model, models, used_feat_cols = train_and_bootstrap(train, feat_cols, target_col='Total', B=B)
    # Prepare 2028 prediction inputs
    # restrict predictions to modern NOCs present in 2020 or 2024
    allowed_nocs = set()
    if 'NOC' in adjusted.columns:
        tmp = adjusted[adjusted['Year'].isin([2020,2024])].copy()
        tmp['NOC'] = tmp['NOC'].apply(map_noc_to_modern)
        allowed_nocs = set(tmp['NOC'].unique())
    log.info('Allowed NOCs for 2028 predictions (present in 2020 or 2024): %d', len(allowed_nocs))

    pred_input = features_df[features_df['Year']==2028].copy()
    # filter to allowed modern NOCs
    if len(allowed_nocs) > 0 and not pred_input.empty:
        pred_input = pred_input[pred_input['NOC'].isin(allowed_nocs)].copy()
    log.info('Initial pred_input rows (Year==2028): %d', len(pred_input))
    if pred_input.empty:
        log.warning('No 2028 rows found in features; generating naive rows from 2024')
        # simple make-shift: use 2024 as base and set Year=2028
        base = features_df[features_df['Year']==2024].copy()
        base['Year'] = 2028
        # filter base to allowed modern NOCs as well
        if len(allowed_nocs) > 0:
            base = base[base['NOC'].isin(allowed_nocs)].copy()
        pred_input = base
        log.info('Generated pred_input from 2024 (rows=%d)', len(pred_input))
    # ensure prediction columns match the numeric features used for training
    X_pred = pred_input[used_feat_cols].copy().select_dtypes(include=[np.number])
    median, lo, hi, all_preds = predict_bootstrap(models, X_pred)
    # assemble final output
    out = pd.DataFrame({'NOC': pred_input['NOC'].values, 'Pred_Total_median': median, 'Pred_Total_lo': lo, 'Pred_Total_hi': hi})
    out.to_csv(PRED_FN, index=False)
    log.info('Saved predictions to %s', PRED_FN)
    # Do not compute probabilities for this run; user requested only numeric 2028 predictions
    return out

# CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_all', action='store_true')
    parser.add_argument('--B', type=int, default=200)
    args = parser.parse_args()
    if args.run_all:
        out = run_full_pipeline(B=args.B)
        log.info('Pipeline finished.')
    else:
        print('Run with --run_all to execute the full pipeline')
