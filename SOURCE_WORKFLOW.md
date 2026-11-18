# Scientific workflow

The 39 source scripts cover canopy aggregation, Landsat temperature quality and aggregation, analysis-sample construction, spatial cross-fitting, forest estimation, social-group estimation, descriptive distributions and figures. They are provided for methodological inspection alongside the runnable summary workflow.

Source copies replace 14 local machine Path constants with an explicit path configuration. Numerical algorithms, model parameters, seeds and input-validation logic are preserved. Complete upstream execution uses satellite products, intermediate processing outputs, saved model checkpoints and original run metadata described by the source scripts. Configure these inputs before running a particular upstream stage.

`source_config.json` defines the project root and external reference notebook. For an upstream source invocation, add `source_support` to PYTHONPATH and set NC_REVIEW_CONFIG to the absolute path of source_config.json. `run_reproduction.py` configures the separate summary workflow automatically.

Historical experiment audits record econml 0.17.0. The final geographically adjusted model's run metadata does not contain a direct econml version entry. The tested source-import environment uses econml 0.16.0 and is listed in requirements_source_review.txt. The summary workflow requires only NumPy and pandas.

Primary sources: US Forest Service and USGS National Land Cover Database tree canopy products; USGS Landsat; US Census Bureau geographic and demographic products; gridMET meteorological data. Product-specific acquisition and processing settings are recorded in the corresponding scripts. Source-provider terms continue to apply to their products.

The repository is publicly accessible for editorial and peer review. No reuse licence has been assigned; third-party source-provider terms continue to apply.
