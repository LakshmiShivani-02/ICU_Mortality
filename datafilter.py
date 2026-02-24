# Python script to extract and prepare MIMIC-IV data for XAI ICU (Diabetic Cohort, 50k-100k patients)
# Updated: Fixed Counter.update bug by adding .to_dict() for value_counts; Assumes files are unzipped (.csv); Added more debugging
from pathlib import Path
import pandas as pd
from collections import Counter
from itertools import combinations
import networkx as nx
from sklearn.utils import resample
import os
import sys

# === Configuration ===
BASE_DIR = Path("/content/drive/My Drive/IIIT Ranchi/mimic_iv_extracted/mimic-iv-3.1")
HOSP = BASE_DIR / "hosp"
ICU = BASE_DIR / "icu"
TARGET_SIZE = 100000  # or 50000
DESIRED_MORTALITY_RATE = 0.16  # Aim for ~16% (within 15-18%)
CHUNK_SIZE = 500000
SUBSET_DIR = Path(f"/content/data_{TARGET_SIZE//1000}k")
SUBSET_DIR.mkdir(exist_ok=True)

# Mandatory chartevents itemids for validation
MANDATORY_ITEMIDS = [220045, 220179, 220180, 220210, 220277, 223761, 220621, 225664]

# Diabetic medications (lowercase)
DIABETIC_MEDS = ['insulin', 'metformin', 'glipizide', 'glyburide', 'glimepiride']

# Files to extract (with paths; assuming .csv since unzipped)
files_list = [
    (HOSP, "admissions.csv", "admissions"),
    (ICU, "icustays.csv", "icustays"),
    (ICU, "chartevents.csv", "chartevents"),
    (ICU, "inputevents.csv", "inputevents"),
    (ICU, "outputevents.csv", "outputevents"),
    (ICU, "procedureevents.csv", "procedureevents"),
    (HOSP, "drgcodes.csv", "drgcodes"),
    (HOSP, "prescriptions.csv", "prescriptions"),
    (HOSP, "diagnoses_icd.csv", "diagnoses_icd"),
    (HOSP, "patients.csv", "patients"),
]




















# === Step 1: Load and filter cohort (with debugging and chunking) ===
print("Step 1: Building diabetic ICU cohort (age >=18, >=1 ICU stay, diabetic, >=5 chart events)...")

# Helper function to check file existence
def safe_read_csv(path, *args, **kwargs):
    if not path.exists():
        print(f"⚠️ File missing: {path} - Treating as empty DataFrame.")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, *args, **kwargs)
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return pd.DataFrame()

# Patients: age >=18
patients_path = HOSP / "patients.csv"
patients = safe_read_csv(patients_path, usecols=['subject_id', 'anchor_age'])
adult = patients[patients['anchor_age'] >= 18]
print(f"Adult patients (>=18): {len(adult):,}")

# ICU patients
icustays_path = ICU / "icustays.csv"
icustays = safe_read_csv(icustays_path, usecols=['subject_id'])
icu_patients = set(icustays['subject_id'].unique())
print(f"Unique ICU patients: {len(icu_patients):,}")
adult_icu = adult[adult['subject_id'].isin(icu_patients)]
print(f"Adult ICU patients: {len(adult_icu):,}")

# Diabetic from ICD (chunked loading)
diagnoses_path = HOSP / "diagnoses_icd.csv"
diabetic_icd_set = set()
if diagnoses_path.exists():
    print("Loading diagnoses_icd in chunks...")
    chunks_processed = 0
    total_rows_diag = 0
    for chunk in pd.read_csv(diagnoses_path, chunksize=CHUNK_SIZE, low_memory=False):
        chunks_processed += 1
        total_rows_diag += len(chunk)
        filtered = chunk[
            ((chunk['icd_version'] == 10) & chunk['icd_code'].str.startswith(('E10', 'E11', 'E12', 'E13', 'E14'))) |
            ((chunk['icd_version'] == 9) & chunk['icd_code'].str.startswith('250'))
        ]
        diabetic_icd_set.update(filtered['subject_id'].unique())
    print(f"Processed {chunks_processed} chunks, {total_rows_diag:,} rows for diagnoses_icd")
else:
    print("⚠️ diagnoses_icd.csv missing - No ICD-based diabetics.")
print(f"Diabetic patients from ICD: {len(diabetic_icd_set):,}")

# Diabetic from prescriptions (chunked loading)
prescriptions_path = HOSP / "prescriptions.csv"
diabetic_rx_set = set()
if prescriptions_path.exists():
    print("Loading prescriptions in chunks...")
    chunks_processed = 0
    total_rows_rx = 0
    for chunk in pd.read_csv(prescriptions_path, chunksize=CHUNK_SIZE, usecols=['subject_id', 'drug'], low_memory=False):
        chunks_processed += 1
        total_rows_rx += len(chunk)
        chunk['drug'] = chunk['drug'].str.lower()
        filtered = chunk[chunk['drug'].isin(DIABETIC_MEDS)]
        diabetic_rx_set.update(filtered['subject_id'].unique())
    print(f"Processed {chunks_processed} chunks, {total_rows_rx:,} rows for prescriptions")
else:
    print("⚠️ prescriptions.csv missing - No Rx-based diabetics.")
print(f"Diabetic patients from Rx: {len(diabetic_rx_set):,}")

# Combined diabetic
diabetic_patients = diabetic_icd_set | diabetic_rx_set
print(f"Total unique diabetic patients: {len(diabetic_patients):,}")
adult_icu_diabetic = adult_icu[adult_icu['subject_id'].isin(diabetic_patients)]
print(f"Adult ICU diabetic patients: {len(adult_icu_diabetic):,}")

# At least 5 chart events (chunked)
chartevents_path = ICU / "chartevents.csv"
chart_counts = Counter()
total_rows_chart = 0
chunks_processed_chart = 0
unique_subjects_chart = set()
if chartevents_path.exists():
    print("Counting chart events in chunks...")
    for chunk in pd.read_csv(chartevents_path, chunksize=CHUNK_SIZE, usecols=['subject_id'], low_memory=False):
        chunks_processed_chart += 1
        total_rows_chart += len(chunk)
        vc = chunk['subject_id'].value_counts().to_dict()  # FIXED: .to_dict()
        chart_counts.update(vc)
        unique_subjects_chart.update(chunk['subject_id'].unique())
        print(f"Processed chunk {chunks_processed_chart}: {len(chunk):,} rows, cumulative unique subjects: {len(unique_subjects_chart):,}")
else:
    print("⚠️ chartevents.csv missing - Assuming no chart events.")
print(f"Processed {chunks_processed_chart} chunks, {total_rows_chart:,} rows from chartevents")
print(f"Total unique subjects in chartevents: {len(chart_counts):,}")
patients_with_5plus = {sid for sid, cnt in chart_counts.items() if cnt >= 5}
print(f"Patients with >=5 chart events: {len(patients_with_5plus):,}")

# Debug overlaps
diabetic_set = set(adult_icu_diabetic['subject_id'])
overlap = diabetic_set & patients_with_5plus
print(f"Overlap between adult ICU diabetic and patients with >=5 charts: {len(overlap):,}")

if len(overlap) == 0 and len(diabetic_set) > 0 and len(patients_with_5plus) > 0:
    print("Sample ICU diabetic subject_ids:", sorted(list(diabetic_set)[:5]))
    print("Sample chart patients subject_ids:", sorted(list(patients_with_5plus)[:5]))

cohort = adult_icu_diabetic[adult_icu_diabetic['subject_id'].isin(patients_with_5plus)]
print(f"Total eligible diabetic ICU patients: {len(cohort):,}")

if len(cohort) == 0:
    print("❌ No eligible patients found! Check data consistency or file completeness.")
    sys.exit(1)

# === Step 2: Add mortality and stratified sample ===
admissions_path = HOSP / "admissions.csv"
admissions = safe_read_csv(admissions_path, usecols=['subject_id', 'hadm_id', 'hospital_expire_flag'])
mortality = admissions.groupby('subject_id')['hospital_expire_flag'].max()
cohort_with_mort = cohort.set_index('subject_id').join(mortality, how='left').fillna(0).reset_index()
natural_rate = cohort_with_mort['hospital_expire_flag'].mean()
print(f"Natural mortality rate in cohort: {natural_rate:.2%}")

# Stratified sampling
dead = cohort_with_mort[cohort_with_mort['hospital_expire_flag'] == 1]
alive = cohort_with_mort[cohort_with_mort['hospital_expire_flag'] == 0]
print(f"Dead in cohort: {len(dead):,}, Alive: {len(alive):,}")

if len(cohort) <= TARGET_SIZE:
    sampled = cohort_with_mort
    print("Cohort smaller than target - taking all.")
else:
    dead_size = min(len(dead), int(TARGET_SIZE * DESIRED_MORTALITY_RATE))
    alive_size = TARGET_SIZE - dead_size
    dead_sample = resample(dead, n_samples=dead_size, replace=(dead_size > len(dead)), random_state=42)
    alive_sample = resample(alive, n_samples=alive_size, replace=False, random_state=42)
    sampled = pd.concat([dead_sample, alive_sample])

sampled_subjects = set(sampled['subject_id'])
print(f"Sampled {len(sampled):,} patients (mortality rate: {sampled['hospital_expire_flag'].mean():.2%})")

# Save sampled subjects
pd.DataFrame({'subject_id': list(sampled_subjects)}).to_csv(SUBSET_DIR / "sampled_subjects.csv", index=False)














# === Step 3: Extract filtered data for each file ===
print("\nStep 3: Extracting data for sampled patients...")

for dir_path, filename, out_name in files_list:
    full_path = dir_path / filename
    if not full_path.exists():
        print(f"Skipping {out_name} (file not found)")
        continue

    subset_path = SUBSET_DIR / (
        f"{out_name}_{TARGET_SIZE//1000}k.csv"
        if out_name != 'diagnoses_icd'
        else "diagnoses_icd.csv"
    )

    print(f"Processing {out_name.ljust(20)} ... ", end="")

    # Remove old file if exists
    if subset_path.exists():
        subset_path.unlink()

    chunks = pd.read_csv(full_path, chunksize=CHUNK_SIZE, low_memory=False)

    total_rows = 0
    first_write = True  # to control header writing

    for chunk in chunks:
        if 'subject_id' not in chunk.columns:
            continue

        filtered = chunk[chunk['subject_id'].isin(sampled_subjects)]

        if not filtered.empty:
            # Write directly to disk instead of storing in RAM
            filtered.to_csv(
                subset_path,
                mode='w' if first_write else 'a',
                header=first_write,
                index=False
            )
            first_write = False
            total_rows += len(filtered)

        # Free memory explicitly
        del chunk, filtered

    if total_rows > 0:
        print(f"{total_rows:,} rows")
    else:
        print("∅ No data")
print("completed")























# === Step 4: Build ICD Knowledge Graph (from full diagnoses_icd if available) ===
print("\nStep 4: Building ICD Knowledge Graph...")
diagnoses_icd_full = safe_read_csv(HOSP / "diagnoses_icd.csv", usecols=['hadm_id', 'icd_code'])
if len(diagnoses_icd_full) == 0:
    print("⚠️ No diagnoses_icd data - Skipping graph build.")
else:
    unique_icd = set(diagnoses_icd_full['icd_code'].unique())
    G = nx.Graph()
    G.add_nodes_from(unique_icd)

    # Hierarchical prefix edges
    for code in list(unique_icd):
        parent = code[:-1]
        while len(parent) > 0 and parent in unique_icd:
            G.add_edge(code, parent)
            parent = parent[:-1]

    # Co-occurrence edges
    grouped = diagnoses_icd_full.groupby('hadm_id')['icd_code'].apply(set)
    for icd_set in grouped:
        if len(icd_set) > 1:
            for a, b in combinations(sorted(icd_set), 2):
                G.add_edge(a, b)

    nx.write_gml(G, SUBSET_DIR / "icd_graph.gml")
    print("✓ ICD graph saved (nodes:", G.number_of_nodes(), "edges:", G.number_of_edges(), ")")

    # Save icd_code per hadm_id (for sampled)
    sampled_hadm = set(admissions[admissions['subject_id'].isin(sampled_subjects)]['hadm_id'])
    filtered_diagnoses = diagnoses_icd_full[diagnoses_icd_full['hadm_id'].isin(sampled_hadm)]
    hadm_icd = filtered_diagnoses.groupby('hadm_id')['icd_code'].apply(list).reset_index()
    hadm_icd.to_csv(SUBSET_DIR / "hadm_icd.csv", index=False)
    print("✓ hadm_icd.csv saved")

# === Step 5: Validation Checks ===
print("\nStep 5: Validation...")
if len(sampled_subjects) == 0:
    print("❌ No patients sampled - Validation skipped.")
else:
    patients_subset_path = SUBSET_DIR / f"patients_{TARGET_SIZE//1000}k.csv"
    if patients_subset_path.exists():
        patients_subset = pd.read_csv(patients_subset_path)
        total_patients = len(patients_subset)
        print(f"Total patients: {total_patients:,}")
    else:
        print("⚠️ patients subset not found.")

    icustays_subset_path = SUBSET_DIR / f"icustays_{TARGET_SIZE//1000}k.csv"
    if icustays_subset_path.exists():
        icustays_subset = pd.read_csv(icustays_subset_path)
        print(f"ICU stays: {len(icustays_subset):,}")
    else:
        print("⚠️ icustays subset not found.")

    admissions_subset_path = SUBSET_DIR / f"admissions_{TARGET_SIZE//1000}k.csv"
    if admissions_subset_path.exists():
        admissions_subset = pd.read_csv(admissions_subset_path)
        mortality_count = admissions_subset[admissions_subset['hospital_expire_flag'] == 1]['subject_id'].nunique()
        mortality_rate = mortality_count / total_patients * 100 if 'total_patients' in locals() else 0
        print(f"Mortality count: {mortality_count:,} (rate: {mortality_rate:.2f}%)")
    else:
        print("⚠️ admissions subset not found.")

    print(f"Diabetic cohort size: {len(sampled):,} (all sampled are diabetic by criteria)")

    min_charts = min(chart_counts.get(sid, 0) for sid in sampled_subjects) if chart_counts else 0
    print(f"Minimum chart events per patient: {min_charts} (should be >=5)")

    chartevents_subset_path = SUBSET_DIR / f"chartevents_{TARGET_SIZE//1000}k.csv"
    if chartevents_subset_path.exists():
        itemids = set()
        for chunk in pd.read_csv(chartevents_subset_path, chunksize=CHUNK_SIZE, usecols=['itemid']):
            itemids.update(chunk['itemid'].unique())
        missing = set(MANDATORY_ITEMIDS) - itemids
        print(f"Required itemids present: {len(missing) == 0} (missing: {missing})")
    else:
        print("⚠️ chartevents subset not found.")

print("\n🎉 EXTRACTION COMPLETE! Data ready in", SUBSET_DIR)
if os.path.exists(SUBSET_DIR):
    total_size = sum(os.path.getsize(f) for f in SUBSET_DIR.glob('**/*') if f.is_file()) / (1024**3)
    print(f"Total size: {total_size:.1f} GB")
else:
    print("No data extracted.")