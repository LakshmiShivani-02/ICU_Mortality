import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
import json

warnings.filterwarnings('ignore')

DATA_DIR = Path("data100k")
OUTPUT_DIR = Path("eda_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# MIMIC-IV Item IDs for vital signs
VITAL_ITEM_IDS = {
    220045: "Heart Rate",
    220179: "Systolic BP",
    220180: "Diastolic BP",
    220210: "Respiratory Rate",
    220277: "SpO2",
    223761: "Temperature",
    220739: "GCS Eye",
    223900: "GCS Verbal",
    223901: "GCS Motor",
}

# Vasopressor drugs for outcome analysis
VASOPRESSORS = ['norepinephrine', 'epinephrine', 'vasopressin', 
                'dopamine', 'phenylephrine', 'dobutamine']


# ============================================================
# DATA LOADING
# ============================================================

def load_all_data():
    """Load all available MIMIC-IV CSV files from data_10k folder."""
    print("=" * 70)
    print("LOADING MIMIC-IV DATA FILES")
    print("=" * 70)
    
    data = {}
    
    # Define expected files and their table names
    file_mapping = {
        'patients': 'patients_100k.csv',
        'admissions': 'admissions_100k.csv',
        'icustays': 'icustays_100k.csv',
        'chartevents': 'chartevents_100k.csv',
        'prescriptions': 'prescriptions_100k.csv',
        'inputevents': 'inputevents_100k.csv',
        'outputevents': 'outputevents_100k.csv',
        'procedureevents': 'procedureevents_100k.csv',
        'microbiologyevents': 'microbiologyevents_100k.csv',
        'hadm_icd': 'hadm_icd_100k.csv',
        'drgcodes': 'drgcodes_100k.csv'
    }
    
    # Large files that need chunked loading (>1GB)
    large_files = {'chartevents'}
    
    for name, filename in file_mapping.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            try:
                if name in large_files:
                    # Load large files in chunks, sample across file for patient diversity
                    print(f"  Loading {name} in chunks (large file ~34GB)...")
                    chunks = pd.read_csv(filepath, chunksize=100000, low_memory=False)
                    samples = []
                    total_patients = set()
                    target_patients = 5000  # Target number of unique patients
                    max_chunks = 100  # Maximum chunks to scan
                    
                    for i, chunk in enumerate(chunks):
                        # Keep only rows with valid valuenum for chartevents
                        if 'valuenum' in chunk.columns:
                            chunk = chunk.dropna(subset=['valuenum'])
                        
                        # Sample to get diverse patients
                        if 'subject_id' in chunk.columns:
                            new_patients = set(chunk['subject_id'].unique()) - total_patients
                            if new_patients:
                                new_patient_data = chunk[chunk['subject_id'].isin(new_patients)]
                                samples.append(new_patient_data)
                                total_patients.update(new_patients)
                        else:
                            samples.append(chunk.sample(min(10000, len(chunk))))
                        
                        if (i + 1) % 20 == 0:
                            print(f"    Scanned {(i+1)*100000:,} rows, found {len(total_patients):,} patients...")
                        
                        if len(total_patients) >= target_patients or i >= max_chunks:
                            break
                    
                    df = pd.concat(samples, ignore_index=True)
                    n_patients = df['subject_id'].nunique() if 'subject_id' in df.columns else 0
                    print(f"✓ {name:20s}: {len(df):>12,} rows | {n_patients:,} patients sampled")
                else:
                    df = pd.read_csv(filepath, low_memory=False)
                    size_mb = df.memory_usage(deep=True).sum() / 1e6
                    print(f"✓ {name:20s}: {len(df):>12,} rows | {len(df.columns):>3} cols | {size_mb:>7.1f} MB")
                data[name] = df
            except Exception as e:
                print(f"✗ {filename}: Error loading - {e}")
        else:
            print(f"· {filename}: Not found (skipping)")
    
    print(f"\n📊 Total tables loaded: {len(data)}")
    return data


# ============================================================
# PATIENT COHORT ANALYSIS
# ============================================================

def analyze_patient_cohort(data):
    """Analyze patient demographics and admission characteristics."""
    print("\n" + "=" * 70)
    print("PATIENT COHORT ANALYSIS")
    print("=" * 70)
    
    if 'patients' not in data or 'admissions' not in data:
        print("⚠ Missing patients or admissions data")
        return {}
    
    patients = data['patients']
    admissions = data['admissions']
    
    # Basic demographics
    n_patients = patients['subject_id'].nunique()
    n_admissions = admissions['hadm_id'].nunique()
    
    print(f"\n📊 Demographics:")
    print(f"   • Unique Patients: {n_patients:,}")
    print(f"   • Total Admissions: {n_admissions:,}")
    print(f"   • Admissions per Patient: {n_admissions/n_patients:.2f}")
    
    # Age distribution
    if 'anchor_age' in patients.columns:
        age_stats = patients['anchor_age'].describe()
        print(f"\n📊 Age Distribution:")
        print(f"   • Mean: {age_stats['mean']:.1f} years")
        print(f"   • Std: {age_stats['std']:.1f} years")
        print(f"   • Min: {age_stats['min']:.0f}, Max: {age_stats['max']:.0f}")
    
    # Gender distribution
    if 'gender' in patients.columns:
        gender_dist = patients['gender'].value_counts(normalize=True)
        print(f"\n📊 Gender Distribution:")
        for gender, pct in gender_dist.items():
            print(f"   • {gender}: {pct:.1%}")
    
    # Mortality analysis
    if 'hospital_expire_flag' in admissions.columns:
        mortality_rate = admissions['hospital_expire_flag'].mean()
        print(f"\n📊 Mortality:")
        print(f"   • In-hospital mortality rate: {mortality_rate:.1%}")
    elif 'deathtime' in admissions.columns:
        mortality_count = admissions['deathtime'].notna().sum()
        mortality_rate = mortality_count / len(admissions)
        print(f"\n📊 Mortality:")
        print(f"   • Deaths: {mortality_count:,} ({mortality_rate:.1%})")
    else:
        mortality_rate = None
    
    # Length of stay
    if 'admittime' in admissions.columns and 'dischtime' in admissions.columns:
        admissions['admittime'] = pd.to_datetime(admissions['admittime'])
        admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
        admissions['los_hours'] = (admissions['dischtime'] - admissions['admittime']).dt.total_seconds() / 3600
        
        print(f"\n📊 Length of Stay:")
        print(f"   • Median: {admissions['los_hours'].median():.1f} hours ({admissions['los_hours'].median()/24:.1f} days)")
        print(f"   • Mean: {admissions['los_hours'].mean():.1f} hours ({admissions['los_hours'].mean()/24:.1f} days)")
    
    # ICU stays
    if 'icustays' in data:
        icu = data['icustays']
        n_icu_stays = len(icu)
        icu_patients = icu['subject_id'].nunique()
        print(f"\n📊 ICU Stays:")
        print(f"   • Total ICU stays: {n_icu_stays:,}")
        print(f"   • Patients with ICU stay: {icu_patients:,} ({icu_patients/n_patients:.1%})")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Age distribution
    if 'anchor_age' in patients.columns:
        sns.histplot(patients['anchor_age'], bins=30, ax=axes[0, 0], color='#3498db')
        axes[0, 0].axvline(patients['anchor_age'].median(), color='red', linestyle='--', 
                           label=f'Median: {patients["anchor_age"].median():.0f}')
        axes[0, 0].set_xlabel('Age (years)', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title('Patient Age Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
    
    # 2. Gender distribution
    if 'gender' in patients.columns:
        gender_counts = patients['gender'].value_counts()
        colors = ['#3498db', '#e74c3c']
        axes[0, 1].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
                       colors=colors, explode=[0.05]*len(gender_counts))
        axes[0, 1].set_title('Gender Distribution', fontsize=14, fontweight='bold')
    
    # 3. Length of stay distribution
    if 'los_hours' in admissions.columns:
        los_days = admissions['los_hours'] / 24
        los_clipped = los_days.clip(upper=los_days.quantile(0.95))
        sns.histplot(los_clipped, bins=40, ax=axes[1, 0], color='#27ae60')
        axes[1, 0].axvline(los_days.median(), color='red', linestyle='--',
                          label=f'Median: {los_days.median():.1f} days')
        axes[1, 0].set_xlabel('Length of Stay (days)', fontsize=12)
        axes[1, 0].set_ylabel('Count', fontsize=12)
        axes[1, 0].set_title('Hospital LOS Distribution (95th percentile)', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
    
    # 4. Mortality pie chart
    if mortality_rate is not None:
        mort_data = [1 - mortality_rate, mortality_rate]
        colors = ['#2ecc71', '#e74c3c']
        axes[1, 1].pie(mort_data, labels=['Survived', 'Deceased'], autopct='%1.1f%%',
                       colors=colors, explode=[0, 0.1])
        axes[1, 1].set_title('In-Hospital Mortality', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cohort_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {OUTPUT_DIR / 'cohort_analysis.png'}")
    
    return {
        'n_patients': n_patients,
        'n_admissions': n_admissions,
        'mortality_rate': mortality_rate if mortality_rate else 0,
        'median_los_hours': admissions['los_hours'].median() if 'los_hours' in admissions.columns else 0
    }


# ============================================================
# VITAL SIGNS ANALYSIS
# ============================================================

def analyze_vitals(data):
    """Analyze vital signs from chartevents."""
    print("\n" + "=" * 70)
    print("VITAL SIGNS ANALYSIS")
    print("=" * 70)
    
    if 'chartevents' not in data:
        print("⚠ chartevents not available")
        return {}
    
    charts = data['chartevents']
    
    # Basic statistics
    n_records = len(charts)
    n_patients = charts['subject_id'].nunique()
    
    print(f"\n📊 Chart Events Overview:")
    print(f"   • Total Records: {n_records:,}")
    print(f"   • Unique Patients: {n_patients:,}")
    print(f"   • Records per Patient: {n_records/n_patients:.0f} (mean)")
    
    # Item ID distribution
    if 'itemid' in charts.columns:
        item_counts = charts['itemid'].value_counts()
        print(f"\n📊 Unique Item Types: {len(item_counts):,}")
        
        # Identify vital signs
        vital_items = charts[charts['itemid'].isin(VITAL_ITEM_IDS.keys())]
        print(f"   • Vital Sign Records: {len(vital_items):,} ({len(vital_items)/n_records:.1%})")
        
        # Top items
        print(f"\n📋 Top 10 Chart Items:")
        for itemid, count in item_counts.head(10).items():
            name = VITAL_ITEM_IDS.get(itemid, f"Item {itemid}")
            print(f"      • {itemid}: {count:,} ({name})")
    
    # Value distribution
    if 'valuenum' in charts.columns:
        value_stats = charts['valuenum'].describe()
        missing_rate = charts['valuenum'].isna().mean()
        print(f"\n📊 Value Statistics:")
        print(f"   • Missing values: {missing_rate:.1%}")
        print(f"   • Mean: {value_stats['mean']:.2f}")
        print(f"   • Std: {value_stats['std']:.2f}")
    
    # Time analysis
    if 'charttime' in charts.columns:
        charts['charttime'] = pd.to_datetime(charts['charttime'], errors='coerce')
        
        # Measurements per patient over time
        patient_counts = charts.groupby('subject_id').size()
        print(f"\n📊 Measurements per Patient:")
        print(f"   • Median: {patient_counts.median():.0f}")
        print(f"   • 25th percentile: {patient_counts.quantile(0.25):.0f}")
        print(f"   • 75th percentile: {patient_counts.quantile(0.75):.0f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Measurements per patient
    sns.histplot(patient_counts.clip(upper=patient_counts.quantile(0.95)), 
                 bins=50, ax=axes[0, 0], color='#3498db')
    axes[0, 0].axvline(patient_counts.median(), color='red', linestyle='--',
                       label=f'Median: {patient_counts.median():.0f}')
    axes[0, 0].set_xlabel('Number of Measurements', fontsize=12)
    axes[0, 0].set_ylabel('Number of Patients', fontsize=12)
    axes[0, 0].set_title('Chart Measurements per Patient', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    
    # 2. Top item types
    top_items = item_counts.head(15)
    labels = [f"{VITAL_ITEM_IDS.get(i, i)}" for i in top_items.index]
    colors = sns.color_palette("husl", len(top_items))
    axes[0, 1].barh(range(len(top_items)), top_items.values, color=colors)
    axes[0, 1].set_yticks(range(len(top_items)))
    axes[0, 1].set_yticklabels(labels, fontsize=9)
    axes[0, 1].set_xlabel('Count', fontsize=12)
    axes[0, 1].set_title('Top 15 Chart Item Types', fontsize=14, fontweight='bold')
    axes[0, 1].invert_yaxis()
    
    # 3. Value distribution (for common vital signs)
    if 'valuenum' in charts.columns and 'itemid' in charts.columns:
        # Heart rate (220045)
        hr_data = charts[charts['itemid'] == 220045]['valuenum'].dropna()
        hr_clipped = hr_data.clip(lower=20, upper=200)
        if len(hr_clipped) > 0:
            sns.histplot(hr_clipped, bins=50, ax=axes[1, 0], color='#e74c3c')
            axes[1, 0].axvline(hr_clipped.median(), color='blue', linestyle='--',
                               label=f'Median: {hr_clipped.median():.0f}')
            axes[1, 0].set_xlabel('Heart Rate (bpm)', fontsize=12)
            axes[1, 0].set_ylabel('Count', fontsize=12)
            axes[1, 0].set_title('Heart Rate Distribution', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
    
    # 4. SpO2 distribution
    if 'valuenum' in charts.columns and 'itemid' in charts.columns:
        spo2_data = charts[charts['itemid'] == 220277]['valuenum'].dropna()
        spo2_clipped = spo2_data.clip(lower=50, upper=100)
        if len(spo2_clipped) > 0:
            sns.histplot(spo2_clipped, bins=50, ax=axes[1, 1], color='#27ae60')
            axes[1, 1].axvline(spo2_clipped.median(), color='blue', linestyle='--',
                               label=f'Median: {spo2_clipped.median():.1f}')
            axes[1, 1].set_xlabel('SpO2 (%)', fontsize=12)
            axes[1, 1].set_ylabel('Count', fontsize=12)
            axes[1, 1].set_title('SpO2 Distribution', fontsize=14, fontweight='bold')
            axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'vitals_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {OUTPUT_DIR / 'vitals_analysis.png'}")
    
    return {
        'n_records': n_records,
        'n_patients': n_patients,
        'records_per_patient': n_records / n_patients,
        'missing_rate': missing_rate if 'valuenum' in charts.columns else 0
    }


# ============================================================
# TIME GAP ANALYSIS
# ============================================================

def analyze_time_gaps(data):
    """Analyze temporal gaps between events (critical for Digital Twin)."""
    print("\n" + "=" * 70)
    print("TIME GAP ANALYSIS (Δt)")
    print("=" * 70)
    
    if 'chartevents' not in data:
        print("⚠ chartevents not available")
        return {}
    
    charts = data['chartevents'].copy()
    
    # Parse times and sort
    charts['charttime'] = pd.to_datetime(charts['charttime'], errors='coerce')
    charts = charts.dropna(subset=['charttime'])
    charts = charts.sort_values(['subject_id', 'charttime'])
    
    # Calculate time gaps
    charts['prev_time'] = charts.groupby('subject_id')['charttime'].shift(1)
    charts['delta_t_minutes'] = (charts['charttime'] - charts['prev_time']).dt.total_seconds() / 60
    charts = charts.dropna(subset=['delta_t_minutes'])
    
    # Filter reasonable gaps (< 48 hours)
    delta_t = charts['delta_t_minutes']
    delta_t_filtered = delta_t[delta_t < 48*60]  # < 48 hours
    
    print(f"\n📊 Time Gap Statistics (minutes):")
    print(f"   • Total gaps analyzed: {len(delta_t_filtered):,}")
    print(f"   • Mean Δt: {delta_t_filtered.mean():.1f} min")
    print(f"   • Median Δt: {delta_t_filtered.median():.1f} min")
    print(f"   • Std Δt: {delta_t_filtered.std():.1f} min")
    print(f"   • 5th percentile: {delta_t_filtered.quantile(0.05):.1f} min")
    print(f"   • 25th percentile: {delta_t_filtered.quantile(0.25):.1f} min")
    print(f"   • 75th percentile: {delta_t_filtered.quantile(0.75):.1f} min")
    print(f"   • 95th percentile: {delta_t_filtered.quantile(0.95):.1f} min")
    
    # Analyze gaps by hour of day
    charts['hour'] = charts['charttime'].dt.hour
    hourly_gaps = charts.groupby('hour')['delta_t_minutes'].median()
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Time gap distribution (linear)
    clipped_gaps = delta_t_filtered.clip(upper=delta_t_filtered.quantile(0.95))
    sns.histplot(clipped_gaps, bins=50, ax=axes[0, 0], color='#3498db')
    axes[0, 0].axvline(delta_t_filtered.median(), color='red', linestyle='--',
                       label=f'Median: {delta_t_filtered.median():.1f} min')
    axes[0, 0].set_xlabel('Time Gap (minutes)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Time Gap Distribution (95th percentile)', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    
    # 2. Log-transformed distribution
    log_gaps = np.log1p(delta_t_filtered)
    sns.histplot(log_gaps, bins=50, ax=axes[0, 1], color='#9b59b6')
    axes[0, 1].set_xlabel('Log(1 + Δt in minutes)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Log-Transformed Time Gap Distribution', fontsize=14, fontweight='bold')
    
    # 3. Hourly pattern
    axes[1, 0].bar(hourly_gaps.index, hourly_gaps.values, color='#27ae60')
    axes[1, 0].set_xlabel('Hour of Day', fontsize=12)
    axes[1, 0].set_ylabel('Median Time Gap (min)', fontsize=12)
    axes[1, 0].set_title('Median Time Gap by Hour of Day', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(range(0, 24, 3))
    
    # 4. Gap category distribution
    bins = [0, 5, 15, 30, 60, 120, 360, np.inf]
    labels = ['<5min', '5-15min', '15-30min', '30-60min', '1-2hr', '2-6hr', '>6hr']
    gap_categories = pd.cut(delta_t_filtered, bins=bins, labels=labels)
    category_counts = gap_categories.value_counts().sort_index()
    
    colors = sns.color_palette("husl", len(category_counts))
    axes[1, 1].bar(category_counts.index, category_counts.values, color=colors)
    axes[1, 1].set_xlabel('Time Gap Category', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title('Time Gap Categories', fontsize=14, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'time_gap_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {OUTPUT_DIR / 'time_gap_analysis.png'}")
    
    return {
        'mean_delta_t': delta_t_filtered.mean(),
        'median_delta_t': delta_t_filtered.median(),
        'std_delta_t': delta_t_filtered.std()
    }


# ============================================================
# MEDICATIONS ANALYSIS
# ============================================================

def analyze_medications(data):
    """Analyze medication patterns from prescriptions."""
    print("\n" + "=" * 70)
    print("MEDICATIONS ANALYSIS")
    print("=" * 70)
    
    if 'prescriptions' not in data:
        print("⚠ prescriptions not available")
        return {}
    
    rx = data['prescriptions']
    
    # Basic stats
    n_records = len(rx)
    n_patients = rx['subject_id'].nunique()
    
    print(f"\n📊 Prescription Overview:")
    print(f"   • Total Prescriptions: {n_records:,}")
    print(f"   • Unique Patients: {n_patients:,}")
    print(f"   • Prescriptions per Patient: {n_records/n_patients:.1f}")
    
    # Drug analysis
    if 'drug' in rx.columns:
        unique_drugs = rx['drug'].nunique()
        print(f"   • Unique Drugs: {unique_drugs:,}")
        
        # Top drugs
        top_drugs = rx['drug'].value_counts().head(15)
        print(f"\n📋 Top 15 Prescribed Drugs:")
        for drug, count in top_drugs.items():
            print(f"      • {drug[:40]:40s}: {count:,}")
        
        # Vasopressor analysis
        rx['drug_lower'] = rx['drug'].str.lower().fillna('')
        vaso_mask = rx['drug_lower'].apply(lambda x: any(v in x for v in VASOPRESSORS))
        vaso_patients = rx[vaso_mask]['subject_id'].nunique()
        print(f"\n📊 Vasopressor Usage:")
        print(f"   • Patients receiving vasopressors: {vaso_patients:,} ({vaso_patients/n_patients:.1%})")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Prescriptions per patient
    rx_per_patient = rx.groupby('subject_id').size()
    sns.histplot(rx_per_patient.clip(upper=rx_per_patient.quantile(0.95)), 
                 bins=40, ax=axes[0], color='#f39c12')
    axes[0].axvline(rx_per_patient.median(), color='red', linestyle='--',
                    label=f'Median: {rx_per_patient.median():.0f}')
    axes[0].set_xlabel('Number of Prescriptions', fontsize=12)
    axes[0].set_ylabel('Number of Patients', fontsize=12)
    axes[0].set_title('Prescriptions per Patient', fontsize=14, fontweight='bold')
    axes[0].legend()
    
    # 2. Top drugs
    if 'drug' in rx.columns:
        top_drugs_plot = top_drugs.head(10)
        colors = sns.color_palette("husl", len(top_drugs_plot))
        axes[1].barh(range(len(top_drugs_plot)), top_drugs_plot.values, color=colors)
        axes[1].set_yticks(range(len(top_drugs_plot)))
        axes[1].set_yticklabels([d[:30] for d in top_drugs_plot.index], fontsize=9)
        axes[1].set_xlabel('Count', fontsize=12)
        axes[1].set_title('Top 10 Prescribed Drugs', fontsize=14, fontweight='bold')
        axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'medications_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {OUTPUT_DIR / 'medications_analysis.png'}")
    
    return {
        'n_prescriptions': n_records,
        'n_patients': n_patients,
        'unique_drugs': unique_drugs if 'drug' in rx.columns else 0,
        'vasopressor_patients': vaso_patients if 'drug' in rx.columns else 0
    }


# ============================================================
# MISSINGNESS ANALYSIS
# ============================================================

def analyze_missingness(data):
    """Analyze missing data patterns across all tables."""
    print("\n" + "=" * 70)
    print("MISSINGNESS ANALYSIS")
    print("=" * 70)
    
    missing_stats = []
    
    for name, df in data.items():
        n_rows = len(df)
        n_cols = len(df.columns)
        total_values = n_rows * n_cols
        missing_values = df.isna().sum().sum()
        missing_rate = missing_values / total_values if total_values > 0 else 0
        
        missing_stats.append({
            'Table': name,
            'Rows': n_rows,
            'Columns': n_cols,
            'Missing Values': missing_values,
            'Missing Rate': f"{missing_rate:.2%}"
        })
    
    # Print summary
    missing_df = pd.DataFrame(missing_stats)
    print("\n📊 Missing Data Summary:")
    print(missing_df.to_string(index=False))
    
    # Detailed analysis for key tables
    for table in ['chartevents', 'prescriptions']:
        if table in data:
            df = data[table]
            print(f"\n📋 {table} Column Missingness:")
            col_missing = df.isna().mean().sort_values(ascending=False)
            for col, rate in col_missing.head(10).items():
                if rate > 0:
                    print(f"      • {col}: {rate:.1%}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    tables = [s['Table'] for s in missing_stats]
    rates = [float(s['Missing Rate'].replace('%', ''))/100 for s in missing_stats]
    
    colors = ['#e74c3c' if r > 0.3 else '#f39c12' if r > 0.1 else '#2ecc71' for r in rates]
    bars = ax.bar(tables, rates, color=colors)
    
    ax.set_ylabel('Missing Rate', fontsize=12)
    ax.set_title('Missing Value Rates by Table', fontsize=14, fontweight='bold')
    ax.set_xticklabels(tables, rotation=45, ha='right')
    ax.set_ylim(0, min(1, max(rates) * 1.2) if rates else 1)
    
    for bar, rate in zip(bars, rates):
        if rate > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'missingness_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {OUTPUT_DIR / 'missingness_analysis.png'}")
    
    return missing_stats


# ============================================================
# SUMMARY DASHBOARD
# ============================================================

def create_summary_dashboard(data, cohort_stats, vitals_stats, time_gap_stats, med_stats):
    """Create comprehensive summary dashboard."""
    print("\n" + "=" * 70)
    print("CREATING SUMMARY DASHBOARD")
    print("=" * 70)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Key Statistics Box
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.axis('off')
    stats_text = f"""
    ╔═══════════════════════════════════╗
    ║      MIMIC-IV EDA SUMMARY         ║
    ╠═══════════════════════════════════╣
    ║ Patients: {cohort_stats.get('n_patients', 0):>6,}                 ║
    ║ Admissions: {cohort_stats.get('n_admissions', 0):>6,}               ║
    ║ Mortality: {cohort_stats.get('mortality_rate', 0)*100:>5.1f}%                 ║
    ║                                   ║
    ║ Chart Records: {vitals_stats.get('n_records', 0):>10,}       ║
    ║ Prescriptions: {med_stats.get('n_prescriptions', 0):>10,}       ║
    ║                                   ║
    ║ Median Δt: {time_gap_stats.get('median_delta_t', 0):>6.1f} min           ║
    ║ Mean Δt:   {time_gap_stats.get('mean_delta_t', 0):>6.1f} min           ║
    ╚═══════════════════════════════════╝
    """
    ax1.text(0.1, 0.5, stats_text, fontsize=11, fontfamily='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 2. Mortality pie
    ax2 = fig.add_subplot(2, 3, 2)
    if cohort_stats.get('mortality_rate', 0) > 0:
        mort_rate = cohort_stats['mortality_rate']
        ax2.pie([1-mort_rate, mort_rate], labels=['Survived', 'Deceased'],
                autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], explode=[0, 0.1])
    ax2.set_title('Mortality Distribution', fontweight='bold')
    
    # 3. Data volume by table
    ax3 = fig.add_subplot(2, 3, 3)
    table_sizes = {name: len(df) for name, df in data.items()}
    top_tables = dict(sorted(table_sizes.items(), key=lambda x: x[1], reverse=True)[:8])
    colors = sns.color_palette("husl", len(top_tables))
    ax3.bar(top_tables.keys(), top_tables.values(), color=colors)
    ax3.set_ylabel('Row Count')
    ax3.set_title('Data Volume by Table', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_yscale('log')
    
    # 4. Time gap distribution
    ax4 = fig.add_subplot(2, 3, 4)
    if 'chartevents' in data:
        charts = data['chartevents'].copy()
        charts['charttime'] = pd.to_datetime(charts['charttime'], errors='coerce')
        charts = charts.dropna(subset=['charttime']).sort_values(['subject_id', 'charttime'])
        charts['prev_time'] = charts.groupby('subject_id')['charttime'].shift(1)
        charts['delta_t'] = (charts['charttime'] - charts['prev_time']).dt.total_seconds() / 60
        delta_t = charts['delta_t'].dropna()
        delta_t_clipped = delta_t[delta_t < 360].clip(upper=120)
        sns.histplot(delta_t_clipped, bins=40, ax=ax4, color='#3498db')
        ax4.axvline(delta_t.median(), color='red', linestyle='--', 
                    label=f'Median: {delta_t.median():.1f} min')
        ax4.set_xlabel('Time Gap (minutes)')
        ax4.set_title('Time Gap Distribution', fontweight='bold')
        ax4.legend()
    
    # 5. Patient data coverage
    ax5 = fig.add_subplot(2, 3, 5)
    if 'patients' in data:
        patients = data['patients']
        admits = data.get('admissions', pd.DataFrame())
        icu = data.get('icustays', pd.DataFrame())
        
        categories = ['All Patients', 'With Admissions', 'With ICU Stay']
        counts = [
            len(patients),
            admits['subject_id'].nunique() if not admits.empty else 0,
            icu['subject_id'].nunique() if not icu.empty else 0
        ]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        ax5.bar(categories, counts, color=colors)
        ax5.set_ylabel('Count')
        ax5.set_title('Patient Coverage', fontweight='bold')
    
    # 6. Clinical implications
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    implications = f"""
    ╔═══════════════════════════════════════╗
    ║    CLINICAL AI IMPLICATIONS           ║
    ╠═══════════════════════════════════════╣
    ║                                       ║
    ║ ✓ Irregular time gaps (Δt) support    ║
    ║   Liquid Neural Network approach      ║
    ║                                       ║
    ║ ✓ Rich vital signs data enables       ║
    ║   Digital Twin trajectory modeling    ║
    ║                                       ║
    ║ ✓ Medication data supports Safety     ║
    ║   Layer rule development              ║
    ║                                       ║
    ║ ✓ Class imbalance requires careful    ║
    ║   metric selection (AUC-PR, F1)       ║
    ║                                       ║
    ╚═══════════════════════════════════════╝
    """
    ax6.text(0.1, 0.5, implications, fontsize=10, fontfamily='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.suptitle('MIMIC-IV Clinical AI System - EDA Summary', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / 'summary_dashboard.png'}")


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_eda_report(cohort_stats, vitals_stats, time_gap_stats, med_stats, missing_stats):
    """Generate comprehensive markdown EDA report."""
    
    report = f"""# MIMIC-IV Clinical AI System - EDA Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents an exploratory data analysis of the MIMIC-IV dataset 
for the Clinical AI System implementing Phase 1 (Digital Twin Sandbox) 
and Phase 2 (Safety Layer).

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total Patients | {cohort_stats.get('n_patients', 'N/A'):,} |
| Total Admissions | {cohort_stats.get('n_admissions', 'N/A'):,} |
| In-Hospital Mortality | {cohort_stats.get('mortality_rate', 0)*100:.1f}% |
| Median LOS | {cohort_stats.get('median_los_hours', 0)/24:.1f} days |

---

## Clinical Data Volume

| Data Type | Records | Per Patient |
|-----------|---------|-------------|
| Chart Events | {vitals_stats.get('n_records', 0):,} | {vitals_stats.get('records_per_patient', 0):.0f} |
| Prescriptions | {med_stats.get('n_prescriptions', 0):,} | {med_stats.get('n_prescriptions', 0)/max(med_stats.get('n_patients', 1), 1):.0f} |

---

## Time Gap Analysis (Critical for Digital Twin)

The irregular time intervals between clinical measurements are essential for 
the Liquid Neural Network architecture used in the Digital Twin model.

| Statistic | Value |
|-----------|-------|
| Mean Δt | {time_gap_stats.get('mean_delta_t', 0):.1f} minutes |
| Median Δt | {time_gap_stats.get('median_delta_t', 0):.1f} minutes |
| Std Δt | {time_gap_stats.get('std_delta_t', 0):.1f} minutes |

---

## Key Findings

### Phase 1 Implications (Digital Twin)

1. **Irregular Sampling**: High variance in time gaps confirms the need for 
   adaptive time constant modeling (τ adaptation in Liquid cells)

2. **Rich Feature Space**: Comprehensive vital signs and lab data support 
   multi-trajectory uncertainty quantification

3. **Patient Heterogeneity**: Wide variation in measurements per patient 
   requires robust sequence padding/masking

### Phase 2 Implications (Safety Layer)

1. **Medication Data**: {med_stats.get('unique_drugs', 0):,} unique drugs available 
   for contraindication rule mining

2. **Vasopressor Usage**: {med_stats.get('vasopressor_patients', 0):,} patients received 
   vasopressors - useful for outcome definition

3. **Missing Data**: Missingness patterns inform feature engineering and 
   imputation strategies

---

## Visualizations Generated

- `cohort_analysis.png` - Patient demographics and mortality
- `vitals_analysis.png` - Vital signs distributions
- `time_gap_analysis.png` - Temporal gap patterns
- `medications_analysis.png` - Prescription patterns
- `missingness_analysis.png` - Missing data patterns
- `summary_dashboard.png` - Comprehensive overview

---

*Report generated by EDA Analysis Pipeline for Clinical AI System*
"""
    
    with open(OUTPUT_DIR / 'eda_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Report saved: {OUTPUT_DIR / 'eda_report.md'}")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Run complete EDA analysis."""
    print("\n" + "=" * 70)
    print("   MIMIC-IV EDA FOR CLINICAL AI SYSTEM (PHASE 1 & 2)")
    print("=" * 70)
    
    # Load data
    data = load_all_data()
    
    if not data:
        print("\n⚠ ERROR: No data files found in data_10k/")
        return
    
    # Run analyses
    cohort_stats = analyze_patient_cohort(data)
    vitals_stats = analyze_vitals(data)
    time_gap_stats = analyze_time_gaps(data)
    med_stats = analyze_medications(data)
    missing_stats = analyze_missingness(data)
    
    # Create summary
    if all([cohort_stats, vitals_stats, time_gap_stats, med_stats]):
        create_summary_dashboard(data, cohort_stats, vitals_stats, time_gap_stats, med_stats)
        generate_eda_report(cohort_stats, vitals_stats, time_gap_stats, med_stats, missing_stats)
    
    # Save stats to JSON
    all_stats = {
        'cohort': cohort_stats,
        'vitals': vitals_stats,
        'time_gaps': time_gap_stats,
        'medications': med_stats,
        'generated_at': datetime.now().isoformat()
    }
    
    with open(OUTPUT_DIR / 'eda_stats.json', 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("   EDA ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Results saved to: {OUTPUT_DIR.absolute()}")
    print(f"   • cohort_analysis.png")
    print(f"   • vitals_analysis.png")
    print(f"   • time_gap_analysis.png")
    print(f"   • medications_analysis.png")
    print(f"   • missingness_analysis.png")
    print(f"   • summary_dashboard.png")
    print(f"   • eda_report.md")
    print(f"   • eda_stats.json")


if __name__ == "__main__":
    main()
