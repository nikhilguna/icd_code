#!/usr/bin/env python3
"""
Generate realistic mock MIMIC data for testing without access to real dataset.

Creates:
- NOTEEVENTS.csv: Discharge summaries with clinical structure
- DIAGNOSES_ICD.csv: ICD-9 diagnosis codes
- PROCEDURES_ICD.csv: ICD-9 procedure codes
- D_ICD_DIAGNOSES.csv: Code descriptions
- D_ICD_PROCEDURES.csv: Procedure descriptions

Usage:
    python scripts/generate_mock_data.py --num-samples 100 --output mock_data/raw/
"""

import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)


# Clinical note templates with realistic structure
DISCHARGE_TEMPLATES = [
    # Template 1: Cardiac patient
    {
        "diagnoses": ["41071", "4019", "25000", "2724", "41401"],
        "procedures": ["3606", "8856", "3893"],
        "template": """Admission Date: [**{admit_date}**] Discharge Date: [**{discharge_date}**]
Date of Birth: [**{dob}**] Sex: {sex}
Service: CARDIOLOGY

CHIEF COMPLAINT:
Chest pain

HISTORY OF PRESENT ILLNESS:
The patient is a {age}-year-old {gender} with a history of hypertension, type 2 diabetes mellitus, and hyperlipidemia who presented to the emergency department with acute onset substernal chest pain radiating to the left arm. The pain started approximately 2 hours prior to arrival and was associated with diaphoresis, nausea, and shortness of breath. Patient took aspirin at home without relief. Patient denies similar episodes in the past.

PAST MEDICAL HISTORY:
1. Hypertension - diagnosed 10 years ago, on Lisinopril 10mg daily
2. Type 2 Diabetes Mellitus - on Metformin 1000mg twice daily, HbA1c 7.2%
3. Hyperlipidemia - on Atorvastatin 40mg nightly
4. Former smoker - quit 5 years ago, 30 pack-year history
5. Family history of coronary artery disease - father MI at age 58

PHYSICAL EXAM ON ADMISSION:
Vitals: BP 145/90, HR 88, RR 16, T 98.6F, O2 Sat 98% on room air
General: Alert and oriented x3, in mild distress secondary to chest pain
HEENT: Normocephalic, atraumatic, PERRL, EOMI
Neck: No JVD, no carotid bruits, supple
Cardiovascular: Regular rate and rhythm, S1 S2 normal, no murmurs, rubs or gallops
Lungs: Clear to auscultation bilaterally, no wheezes, rales or rhonchi
Abdomen: Soft, non-tender, non-distended, normal bowel sounds, no organomegaly
Extremities: No edema, 2+ distal pulses bilaterally, no cyanosis

PERTINENT RESULTS:
Initial Labs:
- Troponin I: 2.5 ng/mL (elevated, critical)
- CK-MB: 45 ng/mL (elevated)
- WBC: 11.2, Hgb 14.5, Platelets 245
- BNP: 450 pg/mL
- Glucose: 185 mg/dL
- Creatinine: 1.1 mg/dL

EKG: ST elevations in leads II, III, aVF consistent with inferior wall ST-elevation myocardial infarction. Q waves present.

Echocardiogram: EF 45%, hypokinesis of inferior and inferolateral walls, mild mitral regurgitation, no pericardial effusion.

HOSPITAL COURSE:
The patient was diagnosed with acute ST-elevation myocardial infarction and taken emergently to the cardiac catheterization laboratory. Coronary angiography revealed 95% stenosis of the right coronary artery which was successfully treated with drug-eluting stent placement. Post-procedure course was uncomplicated. Patient was monitored in the CCU for 24 hours and subsequently transferred to the cardiac floor. Medications were optimized including dual antiplatelet therapy, high-intensity statin, ACE inhibitor, and beta blocker. Patient received education on cardiac rehabilitation and lifestyle modifications.

DISCHARGE DIAGNOSIS:
1. Acute inferior ST-elevation myocardial infarction
2. Coronary artery disease with successful PCI to RCA
3. Hypertension
4. Type 2 Diabetes Mellitus
5. Hyperlipidemia

DISCHARGE MEDICATIONS:
1. Aspirin 81mg oral daily
2. Clopidogrel 75mg oral daily (continue for 12 months minimum)
3. Metoprolol succinate 50mg oral daily
4. Lisinopril 10mg oral daily
5. Atorvastatin 80mg oral nightly (increased dose)
6. Metformin 1000mg oral twice daily
7. Nitroglycerin 0.4mg sublingual PRN chest pain

DISCHARGE INSTRUCTIONS:
Patient discharged in stable condition. Follow up with cardiology in 1 week for post-MI evaluation. Continue cardiac rehabilitation program. Strict medication adherence crucial. Call 911 immediately if chest pain returns. Advised on low-sodium, heart-healthy diet. Continue smoking cessation.

FOLLOW-UP:
1. Cardiology clinic [**Telephone/Fax**] in 1 week
2. Primary care physician in 2 weeks
3. Cardiac rehabilitation program starting next week
"""
    },
    
    # Template 2: Respiratory patient
    {
        "diagnoses": ["4809", "49390", "49320", "2780", "4280"],
        "procedures": ["9390", "9671", "9604"],
        "template": """Admission Date: [**{admit_date}**] Discharge Date: [**{discharge_date}**]
Date of Birth: [**{dob}**] Sex: {sex}
Service: PULMONARY

CHIEF COMPLAINT:
Shortness of breath and cough

HISTORY OF PRESENT ILLNESS:
This is a {age}-year-old {gender} with a history of moderate persistent asthma and obesity who presented with progressive shortness of breath, productive cough with green sputum, and fever for 3 days. Patient reports increased use of albuterol inhaler without adequate relief. Associated symptoms include fever to 101F at home, chills, pleuritic chest pain with deep inspiration, and decreased appetite. No recent travel or sick contacts. Patient has not been compliant with inhaled corticosteroid therapy.

PAST MEDICAL HISTORY:
1. Asthma - moderate persistent, diagnosed age 25
2. Obesity - BMI 36, long-standing
3. GERD - on proton pump inhibitor
4. Seasonal allergies
5. No prior hospitalizations for asthma exacerbations

PHYSICAL EXAM ON ADMISSION:
Vitals: T 101.2F, BP 122/75, HR 95, RR 24, O2 Sat 91% on room air
General: Moderate respiratory distress, using accessory muscles, speaking in short sentences
HEENT: Pharynx mildly erythematous, no exudates
Neck: No lymphadenopathy, no stridor
Cardiovascular: Tachycardic, regular rhythm, no murmurs
Lungs: Decreased breath sounds right lower lobe, crackles and rhonchi present, diffuse expiratory wheezes
Abdomen: Obese, soft, non-tender, normal bowel sounds
Extremities: No cyanosis, clubbing or edema

PERTINENT RESULTS:
Labs:
- WBC: 15.2 (elevated with left shift)
- Procalcitonin: 1.8 ng/mL (elevated)
- ABG on RA: pH 7.38, pCO2 42, pO2 68 (hypoxemia)

Chest X-ray: Right lower lobe consolidation consistent with pneumonia. No pleural effusion. Hyperinflation suggestive of underlying obstructive lung disease.

Sputum Culture: Streptococcus pneumoniae (pending sensitivities)
Blood Cultures: No growth (final)

HOSPITAL COURSE:
Patient diagnosed with community-acquired pneumonia complicated by asthma exacerbation and acute hypoxemic respiratory failure. Started empirically on ceftriaxone and azithromycin. Received supplemental oxygen via nasal cannula to maintain SpO2 >92%. Treated with systemic corticosteroids (methylprednisolone) and bronchodilators (albuterol and ipratropium nebulizers). Clinical improvement noted by hospital day 2 with decreased work of breathing and improved oxygenation. Transitioned to oral antibiotics on day 3. Discharged on hospital day 4 with oral antibiotics and steroid taper.

DISCHARGE DIAGNOSIS:
1. Community-acquired pneumonia, right lower lobe
2. Acute hypoxemic respiratory failure - resolved
3. Acute asthma exacerbation
4. Obesity

DISCHARGE MEDICATIONS:
1. Levofloxacin 750mg oral daily for 7 days (4 days remaining)
2. Prednisone 40mg oral daily x2 days, then 20mg x3 days, then 10mg x2 days
3. Albuterol inhaler 2 puffs every 4 hours as needed
4. Fluticasone/Salmeterol 250/50mcg inhaled twice daily
5. Omeprazole 20mg oral daily
6. Montelukast 10mg oral nightly

DISCHARGE INSTRUCTIONS:
Complete full antibiotic course. Follow steroid taper as prescribed. Use inhalers regularly as directed - do not skip maintenance inhaler. Follow up with PCP in 1 week for repeat chest x-ray to ensure pneumonia resolution. Return to ED if worsening shortness of breath, fever returns, or unable to maintain oxygen saturation. Smoking cessation counseling provided.

FOLLOW-UP:
1. Primary care physician in 1 week with repeat CXR
2. Pulmonology clinic in 4 weeks for asthma management
"""
    },
    
    # Template 3: Gastrointestinal patient
    {
        "diagnoses": ["5789", "5715", "2762", "2859", "53081"],
        "procedures": ["4513", "5121", "8844"],
        "template": """Admission Date: [**{admit_date}**] Discharge Date: [**{discharge_date}**]
Date of Birth: [**{dob}**] Sex: {sex}
Service: GASTROENTEROLOGY

CHIEF COMPLAINT:
Abdominal pain and melena

HISTORY OF PRESENT ILLNESS:
{age}-year-old {gender} with history of cirrhosis and GERD presents with 2 days of epigastric abdominal pain and black tarry stools. Patient reports progressive weakness, dizziness, and one episode of coffee-ground emesis this morning. Denies NSAID use. Has been taking proton pump inhibitor irregularly. No recent alcohol use. Family brought patient to ED due to concerning symptoms.

PAST MEDICAL HISTORY:
1. Cirrhosis - alcohol-related, Child-Pugh class B
2. Portal hypertension with known esophageal varices (Grade II on EGD 1 year ago)
3. GERD - long-standing
4. Hyponatremia - chronic
5. Anemia - chronic disease
6. Former alcohol use disorder - sober x2 years

PHYSICAL EXAM ON ADMISSION:
Vitals: BP 95/60 (orthostatic), HR 110, RR 18, T 98.0F, O2 Sat 98% RA
General: Pale, appears unwell, oriented but lethargic
HEENT: Pale conjunctiva, dry mucous membranes
Neck: No JVD
Cardiovascular: Tachycardic, regular rhythm, no murmurs
Lungs: Clear to auscultation bilaterally
Abdomen: Distended, mild epigastric tenderness, shifting dullness present (ascites), liver edge palpable, splenomegaly
Extremities: No edema, weak pulses
Neuro: Alert, oriented x3, asterixis absent

PERTINENT RESULTS:
Labs:
- Hemoglobin: 7.2 g/dL (baseline 10-11)
- Hematocrit: 21%
- Platelets: 85 (thrombocytopenia)
- INR: 1.6
- Albumin: 2.8 g/dL
- Bilirubin total: 2.4 mg/dL
- Creatinine: 1.4 mg/dL (baseline 1.0)
- Sodium: 128 mEq/L

EGD Findings: Actively bleeding Grade III esophageal varices. Successful band ligation x4. Gastric varices present but not bleeding. Portal hypertensive gastropathy.

HOSPITAL COURSE:
Patient admitted to ICU with acute upper GI bleeding secondary to bleeding esophageal varices. Aggressive resuscitation with IV crystalloids and packed red blood cells (4 units total). Started on octreotide drip and IV proton pump inhibitor. Emergency EGD performed with successful variceal band ligation. Patient also started on ceftriaxone for spontaneous bacterial peritonitis prophylaxis. Hemoglobin stabilized after transfusion. Transitioned to oral nadolol for secondary prophylaxis. No further bleeding episodes. Patient discharged once hemodynamically stable with close outpatient follow-up arranged.

DISCHARGE DIAGNOSIS:
1. Acute upper gastrointestinal hemorrhage secondary to bleeding esophageal varices
2. Hemorrhagic shock - resolved
3. Acute blood loss anemia requiring transfusion
4. Cirrhosis with portal hypertension, Child-Pugh class B
5. Hyponatremia

DISCHARGE MEDICATIONS:
1. Pantoprazole 40mg oral twice daily
2. Nadolol 20mg oral daily (for variceal prophylaxis)
3. Furosemide 40mg oral daily
4. Spironolactone 100mg oral daily
5. Lactulose 30mL oral three times daily
6. Multivitamin with folate oral daily

DISCHARGE INSTRUCTIONS:
Follow up with GI in 2 weeks for repeat EGD. Continue all medications as prescribed. Monitor for signs of rebleeding (black stools, vomiting blood, dizziness). Strict alcohol abstinence. Low sodium diet. Daily weights. Return to ED immediately if signs of bleeding recur.

FOLLOW-UP:
1. Gastroenterology clinic in 2 weeks with repeat EGD
2. Hepatology clinic in 1 month
"""
    },
]


# Common ICD-9 codes with realistic frequencies and descriptions
ICD9_DIAGNOSES = [
    # Cardiovascular (most common in ICU)
    ("4019", "Hypertension NOS", "Unspecified essential hypertension", 5000),
    ("41071", "AMI anterior wall", "Acute myocardial infarction of anterolateral wall", 1200),
    ("41401", "Cor ath native vess", "Coronary atherosclerosis of native coronary artery", 2500),
    ("42731", "Atrial fibrillation", "Atrial fibrillation", 2200),
    ("4280", "CHF NOS", "Congestive heart failure unspecified", 1800),
    
    # Respiratory
    ("4809", "Pneumonia organism NOS", "Pneumonia organism unspecified", 1500),
    ("49390", "Asthma unspecified", "Asthma unspecified type", 800),
    ("49320", "Chr obst asth w(ac)exac", "Chronic obstructive asthma with acute exacerbation", 600),
    ("51881", "Acute respiratry failure", "Acute respiratory failure", 1100),
    ("4928", "Respiratory failure NEC", "Other respiratory failure", 900),
    
    # Metabolic/Endocrine
    ("25000", "DMII wo cmp nt st uncntr", "Diabetes mellitus without complication", 3000),
    ("2724", "Hyperlipidemia NEC/NOS", "Other and unspecified hyperlipidemia", 2000),
    ("2762", "Acidosis", "Metabolic acidosis", 700),
    ("2780", "Obesity unspecified", "Obesity unspecified", 1300),
    
    # Renal
    ("5849", "Acute kidney failure NOS", "Acute kidney failure unspecified", 1200),
    ("5859", "Chronic kidney dis NOS", "Chronic kidney disease unspecified", 1000),
    
    # Infectious
    ("99592", "Severe sepsis", "Severe sepsis", 800),
    ("0389", "Septicemia NOS", "Unspecified septicemia", 950),
    
    # Hematologic
    ("2851", "Ac posthemorrhag anemia", "Acute posthemorrhagic anemia", 900),
    ("2859", "Anemia NOS", "Anemia unspecified", 1400),
    
    # Gastrointestinal
    ("5789", "GI hemorrhage NOS", "Gastrointestinal hemorrhage unspecified", 800),
    ("5715", "Cirrhosis of liver NOS", "Cirrhosis of liver without alcohol", 700),
    ("53081", "GERD", "Esophageal reflux", 1100),
    
    # Neurological
    ("78039", "Convulsions NEC", "Other convulsions", 600),
    ("43491", "CVA", "Cerebral artery occlusion with infarction", 850),
    
    # Others
    ("5990", "UTI NOS", "Urinary tract infection site not specified", 1600),
    ("V4582", "Status post PTCA", "Percutaneous transluminal coronary angioplasty status", 400),
]

ICD9_PROCEDURES = [
    # Cardiovascular procedures
    ("3606", "Insertion coro art stent", "Insertion of drug-eluting coronary artery stent", 800),
    ("8856", "Coro angio 2 cath", "Coronary angiography using two catheters", 1200),
    ("3893", "Venous cath NEC", "Venous catheterization NEC", 2000),
    ("3713", "IABP", "Insertion of intra-aortic balloon pump", 300),
    
    # Respiratory procedures
    ("9390", "CPAP", "Continuous positive airway pressure", 900),
    ("9671", "Cont inv mec ven <96 hrs", "Continuous mechanical ventilation <96 hours", 1100),
    ("9604", "Insert endotracheal tube", "Insertion of endotracheal tube", 800),
    
    # Gastrointestinal procedures
    ("4513", "EGD with biopsy", "Esophagogastroduodenoscopy with closed biopsy", 600),
    ("5121", "Percutaneous liver bx", "Percutaneous liver biopsy", 200),
    
    # Diagnostic procedures
    ("8844", "Arterial blood gases", "Arterial blood gas measurement", 3000),
    ("8872", "Diagnostic ultrasound", "Diagnostic ultrasound of other sites of thorax", 1500),
    ("8841", "CV stress test", "Cardiovascular stress test", 400),
    
    # Other procedures
    ("9904", "Packed red cell transf", "Transfusion of packed red blood cells", 1000),
    ("3891", "Arterial catheter", "Arterial catheterization", 1200),
]


def generate_random_date(start_year=2020, end_year=2023):
    """Generate random date in YYYY-MM-DD format."""
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


def generate_discharge_summary(hadm_id: int, subject_id: int, template_idx: int) -> Dict:
    """Generate a discharge summary from a template."""
    template = DISCHARGE_TEMPLATES[template_idx % len(DISCHARGE_TEMPLATES)]
    
    age = random.randint(40, 85)
    sex = random.choice(["M", "F"])
    gender = "male" if sex == "M" else "female"
    
    admit_date = generate_random_date()
    # Discharge 3-7 days later
    discharge_offset = random.randint(3, 7)
    admit_parts = admit_date.split("-")
    discharge_day = int(admit_parts[2]) + discharge_offset
    discharge_date = f"{admit_parts[0]}-{admit_parts[1]}-{discharge_day:02d}"
    
    # Birth date should be ~age years before admission
    birth_year = int(admit_parts[0]) - age
    dob = f"{birth_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    
    text = template["template"].format(
        admit_date=admit_date,
        discharge_date=discharge_date,
        dob=dob,
        sex=sex,
        age=age,
        gender=gender,
    )
    
    return {
        "ROW_ID": hadm_id,
        "SUBJECT_ID": subject_id,
        "HADM_ID": hadm_id,
        "CHARTDATE": discharge_date,
        "CATEGORY": "Discharge summary",
        "TEXT": text,
        "diagnoses": template["diagnoses"],
        "procedures": template["procedures"],
    }


def generate_mock_data(num_samples: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate mock MIMIC data.
    
    Returns:
        (noteevents_df, diagnoses_df, procedures_df)
    """
    print(f"Generating {num_samples} mock clinical records...")
    
    noteevents = []
    diagnoses_records = []
    procedures_records = []
    
    diag_row_id = 1
    proc_row_id = 1
    
    for i in range(num_samples):
        subject_id = 10000 + i
        hadm_id = 100000 + i
        
        # Generate discharge summary
        template_idx = i % len(DISCHARGE_TEMPLATES)
        note = generate_discharge_summary(hadm_id, subject_id, template_idx)
        
        noteevents.append({
            "ROW_ID": note["ROW_ID"],
            "SUBJECT_ID": note["SUBJECT_ID"],
            "HADM_ID": note["HADM_ID"],
            "CHARTDATE": note["CHARTDATE"],
            "CATEGORY": note["CATEGORY"],
            "TEXT": note["TEXT"],
        })
        
        # Add primary diagnoses from template
        primary_diagnoses = note["diagnoses"]
        
        # Add some additional random diagnoses (5-10 total codes per admission)
        num_additional = random.randint(2, 8)
        all_diag_codes = [code for code, _, _, _ in ICD9_DIAGNOSES]
        additional_diagnoses = random.sample(
            [c for c in all_diag_codes if c not in primary_diagnoses],
            min(num_additional, len(all_diag_codes) - len(primary_diagnoses))
        )
        
        all_diagnoses = primary_diagnoses + additional_diagnoses
        
        for seq_num, code in enumerate(all_diagnoses, 1):
            diagnoses_records.append({
                "ROW_ID": diag_row_id,
                "SUBJECT_ID": subject_id,
                "HADM_ID": hadm_id,
                "SEQ_NUM": seq_num,
                "ICD9_CODE": code,
            })
            diag_row_id += 1
        
        # Add procedures from template
        primary_procedures = note["procedures"]
        
        # Maybe add 1-2 more common procedures
        if random.random() > 0.5:
            all_proc_codes = [code for code, _, _, _ in ICD9_PROCEDURES]
            additional_procedures = random.sample(
                [c for c in all_proc_codes if c not in primary_procedures],
                min(random.randint(1, 2), len(all_proc_codes) - len(primary_procedures))
            )
            all_procedures = primary_procedures + additional_procedures
        else:
            all_procedures = primary_procedures
        
        for seq_num, code in enumerate(all_procedures, 1):
            procedures_records.append({
                "ROW_ID": proc_row_id,
                "SUBJECT_ID": subject_id,
                "HADM_ID": hadm_id,
                "SEQ_NUM": seq_num,
                "ICD9_CODE": code,
            })
            proc_row_id += 1
    
    return (
        pd.DataFrame(noteevents),
        pd.DataFrame(diagnoses_records),
        pd.DataFrame(procedures_records),
    )


def create_code_dictionaries() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create D_ICD_DIAGNOSES and D_ICD_PROCEDURES tables."""
    diagnoses_dict = pd.DataFrame([
        {
            "ROW_ID": i + 1,
            "ICD9_CODE": code,
            "SHORT_TITLE": short_title,
            "LONG_TITLE": long_title,
        }
        for i, (code, short_title, long_title, _) in enumerate(ICD9_DIAGNOSES)
    ])
    
    procedures_dict = pd.DataFrame([
        {
            "ROW_ID": i + 1,
            "ICD9_CODE": code,
            "SHORT_TITLE": short_title,
            "LONG_TITLE": long_title,
        }
        for i, (code, short_title, long_title, _) in enumerate(ICD9_PROCEDURES)
    ])
    
    return diagnoses_dict, procedures_dict


def main():
    parser = argparse.ArgumentParser(description="Generate mock MIMIC data for testing")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of discharge summaries to generate (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mock_data/raw/",
        help="Output directory (default: mock_data/raw/)",
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("MOCK MIMIC DATA GENERATOR")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print()
    
    # Generate main tables
    noteevents_df, diagnoses_df, procedures_df = generate_mock_data(args.num_samples)
    
    # Create code dictionaries
    diag_dict_df, proc_dict_df = create_code_dictionaries()
    
    # Save to CSV
    print("Saving files...")
    noteevents_df.to_csv(output_dir / "NOTEEVENTS.csv", index=False)
    print(f"✓ NOTEEVENTS.csv: {len(noteevents_df)} clinical notes")
    
    diagnoses_df.to_csv(output_dir / "DIAGNOSES_ICD.csv", index=False)
    print(f"✓ DIAGNOSES_ICD.csv: {len(diagnoses_df)} diagnosis code assignments")
    
    procedures_df.to_csv(output_dir / "PROCEDURES_ICD.csv", index=False)
    print(f"✓ PROCEDURES_ICD.csv: {len(procedures_df)} procedure code assignments")
    
    diag_dict_df.to_csv(output_dir / "D_ICD_DIAGNOSES.csv", index=False)
    print(f"✓ D_ICD_DIAGNOSES.csv: {len(diag_dict_df)} diagnosis code definitions")
    
    proc_dict_df.to_csv(output_dir / "D_ICD_PROCEDURES.csv", index=False)
    print(f"✓ D_ICD_PROCEDURES.csv: {len(proc_dict_df)} procedure code definitions")
    
    print()
    print("="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Calculate statistics
    unique_patients = noteevents_df["SUBJECT_ID"].nunique()
    unique_admissions = noteevents_df["HADM_ID"].nunique()
    
    codes_per_admission = diagnoses_df.groupby("HADM_ID").size()
    procs_per_admission = procedures_df.groupby("HADM_ID").size()
    
    print(f"Unique patients: {unique_patients}")
    print(f"Unique admissions: {unique_admissions}")
    print(f"Total diagnosis assignments: {len(diagnoses_df)}")
    print(f"Total procedure assignments: {len(procedures_df)}")
    print()
    print(f"Diagnosis codes per admission:")
    print(f"  Mean: {codes_per_admission.mean():.1f}")
    print(f"  Median: {codes_per_admission.median():.0f}")
    print(f"  Range: [{codes_per_admission.min()}, {codes_per_admission.max()}]")
    print()
    print(f"Procedure codes per admission:")
    print(f"  Mean: {procs_per_admission.mean():.1f}")
    print(f"  Median: {procs_per_admission.median():.0f}")
    print(f"  Range: [{procs_per_admission.min()}, {procs_per_admission.max()}]")
    print()
    
    # Most common codes
    print("Top 10 diagnosis codes:")
    top_diag = diagnoses_df["ICD9_CODE"].value_counts().head(10)
    for code, count in top_diag.items():
        code_info = diag_dict_df[diag_dict_df["ICD9_CODE"] == code].iloc[0]
        print(f"  {code}: {code_info['SHORT_TITLE']:30s} ({count} occurrences)")
    
    print()
    print("="*60)
    print("✅ MOCK DATA GENERATION COMPLETE!")
    print("="*60)
    print()
    print("Next steps:")
    print(f"  1. Review generated data in: {output_dir}")
    print(f"  2. Test preprocessing: python scripts/test_preprocessing.py")
    print(f"  3. Test data loading: python scripts/test_data_pipeline.py")
    print()


if __name__ == "__main__":
    main()

