# Dataset Source

## Name
Real or Fake Job Posting Prediction

## Source
- **Platform**: Kaggle
- **URL**: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
- **Author**: Shivam Bansal

## Description
The dataset contains approximately 18,000 job postings, with around 800 (~4.8%) labeled as fraudulent.
Each row has 18 columns covering the job title, location, department, salary range, company profile,
job description, requirements, benefits, employment type, required experience/education/industry/function,
telecommuting flag, logo presence, questions flag, and the binary `fraudulent` target label.

## License
This dataset is shared under the **CC0: Public Domain** license on Kaggle.
It may be freely used for research and educational purposes without restriction.

## Citation
Bansal, S. (2020). Real or Fake? Fake Job Posting Prediction. Kaggle.
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

## Download Instructions
Run:
```bash
python src/download_data.py
```
This uses the `opendatasets` library to fetch the CSV and place it at `data/fake_job_postings.csv`.
