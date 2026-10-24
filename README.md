# Long-term urban canopy change and surface thermal conditions in the United States

This repository provides processed data and analysis code for reviewing the study's long-term canopy distribution, conditional temperature responses and community selection comparisons.

## Reproduce the reported summaries

Use Python 3.11 and install the two replay dependencies:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python run_reproduction.py --output ../review_outputs
```

The three entries reproduce national and social-group estimates, reference intervals and Holm adjustments, figure values and candidate rankings, and resource summaries. They perform 342, 234 and 26 numerical checks. Intervals use the saved joint covariance matrices. Each run writes its numerical results and check records to a new output directory.

## Contents

- `02_source_data/project/`: processed inputs and 39 scientific workflow scripts, retaining project-relative paths.
- `02_source_data/paper_tables/`: manuscript and supplementary tables.
- `03_code/`: the three summary-reproduction entries.
- `DATA_DICTIONARY.csv`: core variables and units.
- `DATA_AND_CODE_INDEX.csv`: file purposes.
- `SOURCE_WORKFLOW.md`: scientific source organization and upstream requirements.

The complete upstream workflow processes remote-sensing data and fits models. The executable summary workflow uses saved predictions, residuals, group probabilities and covariance matrices. Data values, sample definitions and numerical tolerances match the study's archived analyses.

Contact: Zeyang Li, lzy20258384@qlnu.edu.cn.
