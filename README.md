# Social disparities in US urban tree canopy change and temperature responses

This repository provides processed data and analysis code for the study's long-term canopy distribution, conditional temperature responses, measurement comparisons and community selection summaries. The data correspond to the manuscript's fixed 2000 census-tract and urbanized-land definitions.

## Reproduce the reported summaries

Use Python 3.11 and install the two replay dependencies:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python run_reproduction.py --output ../review_outputs
```

The first three entries reproduce national and social-group estimates, reference intervals and Holm adjustments, figure values and candidate rankings, and resource summaries. They perform 342, 234 and 26 deterministic numerical checks. A fourth entry recalculates the canopy-product, local-reference, annual-stage and Figure 5 sensitivity summaries against the current supplementary tables. It performs 635 table-value comparisons. Intervals use the saved joint covariance matrices. Each run writes results to the chosen output directory.

## Contents

- `02_source_data/project/`: processed inputs and 39 scientific workflow scripts.
- `02_source_data/measurement/`: 46,424-tract canopy-product data, annual summaries and matched local-reference data.
- `02_source_data/paper_tables/`: three main tables, three supplementary methods tables and Supplementary Tables 1a–18, exported from the current manuscript and Supplementary Information.
- `03_code/`: four summary-reproduction entries and four upstream measurement scripts.
- `DATA_DICTIONARY.csv`: core variables and units.
- `DATA_AND_CODE_INDEX.csv`: file purposes and sizes.
- `SOURCE_WORKFLOW.md`: scientific source organization and upstream requirements.
- `MEASUREMENT_DATA.md`: analysis units, source data and table correspondence.

The full upstream workflow processes remote-sensing data and fits models. The executable summary workflow uses saved predictions, residuals, group probabilities and covariance matrices. It does not retrain the models or reprocess satellite imagery. The matched local references use their actual observation years and mapped footprints. The 23-city window interpretation record in Supplementary Table 18 is a method diagnostic and does not enter the reported local-reference contrasts.

Contact: Zeyang Li, lzy20258384@qlnu.edu.cn.
