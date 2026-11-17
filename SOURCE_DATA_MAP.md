# Source data for the manuscript

The `02_source_data/figures/` directory contains numerical values used for the six main figures. Each directory is identified by the figure number in the manuscript. Values are supplied at the plotted unit; rendering also uses geographic boundaries, projections and display settings described in the figure legends and scientific source scripts.

| Figure | Panels | Numerical source files |
|---|---|---|
| Fig. 1 | a; b–j | `Figure_1/national_display_points.csv`; `Figure_1/paired_2023_2024_display.csv`; `Figure_1/city_support.csv` |
| Fig. 2 | a–k | `Figure_2/city_net_change_points.csv`; `Figure_2/national_annual_used.csv`; `Figure_2/group_trajectories_used.csv`; `Figure_2/city_gap_change_points.csv`; `Figure_2/equal_city_gaps_used.csv`; `Figure_2/coverage_endpoints_used.csv` |
| Fig. 3 | a–l | `Figure_3/city_paired_means.csv`; `Figure_3/city_levels_and_cumulative_changes.csv`; `Figure_3/social_group_trajectories.csv`; `Figure_3/social_equal_city_gap_trajectories.csv` |
| Fig. 4 | a–h | `Figure_4/map_predictions.csv`; `Figure_4/national_estimates.csv`; `Figure_4/group_means_200km.csv`; `Figure_4/group_contrasts_all_scales.csv` |
| Fig. 5 | a–g | `Figure_5/map_paired_tracts.csv`; `Figure_5/two_dimensional_histogram.csv`; `Figure_5/social_group_composition.csv`; `Figure_5/category_shares_by_scope.csv` |
| Fig. 6 | a–h | `Figure_6/map_states_10percent.csv`; `Figure_6/cutoffs_24_combinations.csv`; `Figure_6/summary_24_combinations.csv` |

Numerical values for Tables 1–3, Supplementary Methods Tables 1–3 and Supplementary Tables 1a–18 are in `02_source_data/paper_tables/`, with one CSV per table. Supplementary measurement comparisons are supported by `02_source_data/measurement/`. Saved predictions, augmented scores, group probabilities and covariance inputs needed for the executable summary workflow are in `02_source_data/project/`.

Fig. 1 and other maps require the public census and urbanized-area boundaries described in the manuscript. These original provider files are not redistributed here. The source data for the two city atlases are derived from the fixed tract panel and the scientific source scripts; the atlas PDFs are part of the Supplementary Information submission.
