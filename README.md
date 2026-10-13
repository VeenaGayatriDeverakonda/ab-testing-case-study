# Checkout Redesign A/B Test: End-to-End Case Study

## Problem

An ecommerce site wants to know whether a redesigned checkout flow actually
lifts purchases. Gut feel says the new flow is cleaner, but shipping on gut
feel is how you break a $71M/year revenue stream. The question is not "did
conversion go up" but "did it go up because of the redesign, by how much,
and is the gain real enough to ship."

## Experiment design

- **Hypothesis**: the redesigned checkout increases visitor-to-purchase
  conversion without hurting average order value.
- **Randomization**: 80,000 users, 50/50 control vs treatment, user-level,
  over 2 weeks (40,000 per arm).
- **Primary metric**: conversion rate (visitor converts to a purchase).
- **Guardrail metrics**: average order value, revenue per visitor.
- **Power analysis**: baseline 12.0%, minimum detectable effect +1.5pp at
  80% power and 5% significance needs 7,761 users per arm. We ran 40,000
  per arm, deliberately overpowered so the device-segment analysis (below)
  has real statistical teeth.

## What I checked before trusting the result

- **Sample ratio mismatch**: exact 40,000/40,000 split, chi-square p = 1.0000.
  Randomization is clean.
- **Peeking/novelty**: week 1 ran a touch hotter (+2.66pp) than week 2
  (+2.17pp), the classic novelty bump, but the effect stays significant and
  stable in week 2. No reason to extend the test.

## Results (measured, n = 80,000)

- **Primary metric**: control 16.71% (6,685/40,000), treatment 19.13%
  (7,650/40,000). Absolute lift **+2.41pp**, 95% CI [+1.88pp, +2.94pp].
  Relative lift **+14.4%**. Two-proportion z-test: z = 8.90, p < 0.0001.
- **Guardrails hold**: average order value $101.18 (control) vs $101.12
  (treatment), p = 0.95, no significant change. Revenue per visitor rose
  from $16.91 to $19.34 (+14.4%), driven entirely by conversion.
- **CUPED variance reduction**: using pre-experiment purchase history as a
  covariate cut estimator variance by 5.1% (CI 2.6% narrower) with the same
  point estimate. Same conclusion, tighter interval.
- **Heterogeneous effects by device**: mobile +2.73pp (95% CI
  [+2.05pp, +3.42pp]), desktop +1.93pp (95% CI [+1.09pp, +2.77pp]).
  Interaction +0.80pp. The redesign helps mobile disproportionately, and
  mobile is 60% of traffic, so the win concentrates where it matters most.
  Desktop is still significantly positive, so ship to everyone.

## Business decision

At 350,000 monthly visitors and $101.18 average order value, a +2.41pp
conversion lift is worth **$854,364/month, or $10.25M annualized** in
incremental revenue.

**Recommendation: SHIP.** Significant lift, guardrails clean, effect durable
into week 2, strongest on the mobile majority of traffic.

## What this project demonstrates

- Experiment design: hypothesis, power analysis, MDE, sample sizing.
- Statistical rigor: z-tests, confidence intervals, SRM checks, CUPED.
- Judgment calls: segment analysis, novelty effects, when to stop a test.
- Business translation: lift in percentage points becomes dollars before
  anyone asks.

## Reproduce it

```bash
python src/generate.py   # builds data/experiment.csv (fixed seed)
python src/analyze.py    # runs the full analysis end to end
```

See `notebooks/01_ab_testing_case_study.ipynb` for the step-by-step walkthrough.
