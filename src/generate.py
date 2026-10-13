"""Generate the synthetic checkout-redesign experiment dataset.

Scenario: an ecommerce site tests a redesigned checkout flow (treatment)
against the existing flow (control). Users are randomized 50/50 at the
user level. The primary metric is conversion (visitor -> purchase) within
the 2-week experiment window.

Fixed seed -> the dataset is fully reproducible. Run:
    python src/generate.py
"""

import numpy as np
import pandas as pd

SEED = 20261013
N_PER_ARM = 40_000          # 40k control + 40k treatment
P_CONTROL = 0.12            # baseline conversion rate
LIFT_MOBILE = 0.020         # +2.0pp treatment effect on mobile
LIFT_DESKTOP = 0.0075       # +0.75pp treatment effect on desktop
P_MOBILE = 0.60             # share of mobile traffic
AOV = 85.0                  # average order value ($), guardrail metric


def main() -> None:
    rng = np.random.default_rng(SEED)

    n = N_PER_ARM * 2
    user_id = np.arange(1, n + 1)
    arm = np.repeat(["control", "treatment"], N_PER_ARM)
    rng.shuffle(arm)
    is_treat = (arm == "treatment").astype(int)

    device = np.where(rng.random(n) < P_MOBILE, "mobile", "desktop")
    is_mobile = (device == "mobile").astype(int)

    # Pre-experiment covariate for CUPED: purchased in the 30 days before.
    # Strongly correlated with outcome (past buyers convert at much higher
    # rates), which is what makes CUPED variance reduction work.
    prior_buyer = (rng.random(n) < 0.25).astype(int)

    # Day of experiment (1-14), for the novelty-effect check.
    day = rng.integers(1, 15, size=n)

    # Conversion probability: baseline + prior-buyer bump + treatment effect
    # (heterogeneous by device) + a small novelty bump for treatment in week 1
    # that fades in week 2.
    novelty = np.where((day <= 7) & (is_treat == 1), 0.004, 0.0)
    p = (
        P_CONTROL
        + 0.20 * prior_buyer
        + is_treat * (LIFT_MOBILE * is_mobile + LIFT_DESKTOP * (1 - is_mobile))
        + novelty
    )
    p = np.clip(p, 0.01, 0.95)
    converted = (rng.random(n) < p).astype(int)

    # Guardrail: average order value, lognormal around $85, no true difference.
    order_value = np.where(
        converted == 1,
        rng.lognormal(mean=np.log(AOV), sigma=0.6, size=n).round(2),
        0.0,
    )

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "arm": arm,
            "device": device,
            "day": day,
            "prior_buyer": prior_buyer,
            "converted": converted,
            "order_value": order_value,
        }
    )
    df.to_csv("data/experiment.csv", index=False)
    print(f"wrote data/experiment.csv: {len(df):,} rows")
    print(df.groupby("arm")["converted"].agg(["mean", "count"]).round(4))


if __name__ == "__main__":
    main()
