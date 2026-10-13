# Data

The dataset is synthetic and fully reproducible. It is not committed to the
repo; generate it with one command (fixed seed, always the same 80,000 rows):

```bash
python src/generate.py
```

This writes `data/experiment.csv` with columns:

| column        | description                                              |
|---------------|----------------------------------------------------------|
| user_id       | unique user identifier                                   |
| arm           | `control` or `treatment` (50/50, user-level randomization)|
| device        | `mobile` or `desktop`                                    |
| day           | day of the 2-week experiment (1-14)                      |
| prior_buyer   | 1 if the user purchased in the 30 days before the test   |
| converted     | 1 if the user purchased during the experiment            |
| order_value   | order value in USD (0 when not converted)                |

Design parameters (in `src/generate.py`, seed 20261013): baseline conversion
12%, treatment effect +2.0pp on mobile and +0.75pp on desktop, 60% mobile
traffic, a small week-1 novelty bump on treatment, and a pre-experiment
purchase covariate for the CUPED section.
