# MIMIC Data Access Guide

## What is MIMIC?

MIMIC (Medical Information Mart for Intensive Care) is a publicly available de-identified health dataset. It contains:
- **MIMIC-III**: 60,000+ ICU admissions (2001-2012) with discharge summaries and ICD-9 codes
- **MIMIC-IV**: Extended dataset (2008-2019) with ICD-9 and ICD-10 codes

⚠️ **MIMIC is restricted data** - it requires credentialed access through PhysioNet because it contains real (de-identified) patient data.

---

## How to Get Access to MIMIC

### Step 1: Complete PhysioNet Credentialing

1. **Go to PhysioNet**: https://physionet.org/

2. **Create an account** (if you don't have one):
   - Click "Sign up"
   - Use your **university email** (e.g., `@umich.edu`)

3. **Complete CITI Training** (Required):
   - PhysioNet requires completion of CITI (Collaborative Institutional Training Initiative) training in "Data or Specimens Only Research"
   - This takes 2-4 hours and is free
   - Link: https://www.citiprogram.org/
   - Search for: "Data or Specimens Only Research"
   - Complete the training and save your completion certificate

4. **Apply for MIMIC-III and/or MIMIC-IV Access**:
   - Go to: https://physionet.org/content/mimiciii/1.4/ (MIMIC-III)
   - Or: https://physionet.org/content/mimiciv/3.0/ (MIMIC-IV)
   - Click "Request access"
   - Upload your CITI completion certificate
   - Wait for approval (usually 1-3 business days)

---

### Step 2: Choose Your Data Access Method

Once credentialed, you have **two options** to access MIMIC data:

#### **Option A: AWS Athena (Recommended for this project)**

MIMIC data is hosted on AWS in the `physionet-data` S3 bucket. You can query it directly using Amazon Athena.

**Prerequisites:**
- ✅ PhysioNet credentials (Step 1)
- ✅ AWS account
- ✅ Your AWS account linked to PhysioNet credentials

**Setup Steps:**

1. **Link your AWS account to PhysioNet:**
   - Log into PhysioNet
   - Go to your profile → "Cloud Storage" or "AWS Access"
   - Request AWS access (PhysioNet will grant permissions)
   - Wait for approval

2. **Verify AWS access:**
   ```bash
   # Test if you can see the MIMIC databases
   aws athena list-databases --catalog-name AwsDataCatalog --region us-east-1
   ```

3. **Find/create an S3 bucket for Athena results:**
   - Go to AWS S3 Console: https://s3.console.aws.amazon.com/
   - Check if there's already a bucket configured for Athena
   - Or ask your instructor/administrator for the bucket name

4. **Configure your project:**
   - Update `configs/default.yaml` with your S3 bucket:
     ```yaml
     data:
       athena_output_bucket: "s3://your-bucket-name/athena-results/"
     ```

5. **Extract data:**
   ```bash
   python scripts/extract_data.py --source athena --dataset mimic3 --output data/raw/mimic3.parquet --bucket s3://your-bucket/athena-results/
   ```

#### **Option B: Download Data Locally**

If AWS Athena isn't available, you can download the data as CSV files.

**Steps:**

1. **Download from PhysioNet:**
   - Go to: https://physionet.org/content/mimiciii/1.4/
   - Click "Files" tab
   - Download the required tables:
     - `NOTEEVENTS.csv.gz` (discharge summaries - **very large, ~5GB compressed**)
     - `DIAGNOSES_ICD.csv.gz` (ICD codes)
   - Decompress the files

2. **Extract using local mode:**
   ```bash
   python scripts/extract_data.py \
     --source local \
     --dataset mimic3 \
     --output data/raw/mimic3.parquet \
     --notes-csv path/to/NOTEEVENTS.csv \
     --diagnoses-csv path/to/DIAGNOSES_ICD.csv
   ```

---

## If You're Starting Fresh (No Access Yet)

### Option 1: Use a Smaller/Public Dataset First

While waiting for MIMIC access approval, you can:

1. **Use a smaller clinical dataset** to test your pipeline:
   - Create a minimal test dataset
   - Verify your preprocessing, models, and training code works

2. **Use synthetic data** for development:
   - Generate fake discharge summaries with ICD codes
   - Test your entire pipeline end-to-end

### Option 2: Contact Your Instructor

Since this is a **CS595 NLP course project**, your instructor may:

- Have pre-approved AWS access set up for students
- Provide a shared S3 bucket for Athena results
- Have alternative ways to access the data
- Have a sample subset you can use for development

**Ask your instructor about:**
- Whether they've set up AWS access for students
- If there's a shared bucket/Athena workgroup
- If you can use a sample dataset while waiting for full access

### Option 3: Check University Resources

- Check if your university has an **Institutional MIMIC license**
- Some universities have shared access through their cloud infrastructure
- Contact your department's IT/research computing team

---

## Timeline Expectations

| Step | Typical Time |
|------|--------------|
| Create PhysioNet account | 5 minutes |
| Complete CITI training | 2-4 hours |
| Submit access request | 10 minutes |
| Wait for approval | 1-3 business days |
| Link AWS account (if needed) | 1-2 business days |
| **Total** | **~1 week** |

**Recommendation:** Start the credentialing process **immediately** if you haven't already!

---

## Troubleshooting

### "AccessDenied" errors in AWS

- Your AWS account may not be linked to PhysioNet yet
- Check your PhysioNet profile for AWS access status
- Contact PhysioNet support if access is pending

### Can't find S3 bucket

- Ask your instructor/administrator for the bucket name
- Check AWS Athena Console → Settings → Query result location
- You may need S3 bucket creation permissions (ask your AWS admin)

### CITI training not accepting

- Make sure you select "Data or Specimens Only Research" (not IRB training)
- Use your university email
- Save the completion certificate as PDF

---

## Next Steps After Getting Access

1. ✅ Verify you can query MIMIC via Athena or access downloaded files
2. ✅ Update `configs/default.yaml` with your S3 bucket (if using Athena)
3. ✅ Run data extraction: `python scripts/extract_data.py ...`
4. ✅ Verify extracted data looks correct
5. ✅ Continue with preprocessing and model training

---

## Useful Links

- **PhysioNet Home**: https://physionet.org/
- **MIMIC-III**: https://physionet.org/content/mimiciii/1.4/
- **MIMIC-IV**: https://physionet.org/content/mimiciv/3.0/
- **CITI Training**: https://www.citiprogram.org/
- **PhysioNet Support**: support@physionet.org
- **MIMIC Documentation**: https://mimic.mit.edu/

---

## Questions?

- **PhysioNet access issues**: support@physionet.org
- **AWS setup**: Check with your instructor or AWS administrator
- **Project-specific**: Ask your CS595 instructor

