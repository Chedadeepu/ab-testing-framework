# A/B Testing & Statistical Hypothesis Testing

**Author:** Saiteja Chedadeepu | [GitHub](https://github.com/Chedadeepu) | [LinkedIn](https://linkedin.com/in/saitejachedadeepu)

> **No dataset download required.** This project generates its own synthetic data — install dependencies and run immediately.

End-to-end A/B testing framework covering experiment design, sample size calculation, statistical significance testing, effect size, confidence intervals, Bonferroni correction for multiple metrics, and a peeking bias demonstration.

---

## Results (Latest Run)

| Metric | Value |
|--------|-------|
| Control CTR | 9.60% |
| Treatment CTR | 11.73% |
| **Absolute lift** | **+2.14 percentage points** |
| **Relative lift** | **+22.3%** |
| z-statistic | −3.03 |
| **p-value** | **0.0024** |
| **95% CI** | **[+0.76pp, +3.52pp]** |
| Cohen's d | 0.069 (small effect) |
| **Decision** | ✅ **REJECT H₀ — Ship the new button!** |

---

## What This Project Covers

| Step | Topic |
|------|-------|
| 1 | Experiment design + **sample size calculation** (NormalIndPower) |
| 2 | Simulating realistic experiment data (numpy binomial) |
| 3 | Summary statistics and descriptive breakdown |
| 4 | **Two-proportions z-test** (primary conversion metric) |
| 5 | **95% Confidence Interval** for the lift |
| 6 | **Cohen's d** — effect size (statistical vs practical significance) |
| 7 | **t-tests** for secondary metrics (session duration, pages viewed) |
| 8 | **Bonferroni correction** for multiple metric testing |
| 9 | **Peeking bias** demonstration with visualisation |
| 10 | Written business report with recommendation |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| NumPy, Pandas | Data simulation & manipulation |
| SciPy | z-test, t-test, statistical functions |
| statsmodels | Sample size (NormalIndPower), proportion_effectsize |
| Matplotlib, Seaborn | Visualisations |

---

## Project Structure

```
ab-testing-framework/
├── ab_test_analysis.py          ← Main script (run this)
├── requirements.txt
├── README.md
├── .gitignore
└── outputs/
    ├── charts/
    │   ├── 01_sample_size_curve.png   ← Effect vs sample size trade-off
    │   ├── 02_ab_test_results.png     ← CTR, CI, session, device breakdown
    │   ├── 03_peeking_bias.png        ← Why you must not stop early
    │   └── 04_decision_report.png     ← Final decision summary card
    ├── ab_experiment_data.csv         ← Simulated user-level data
    └── ab_report.txt                  ← Full written analysis report
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Chedadeepu/ab-testing-framework.git
cd ab-testing-framework

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run — no dataset needed, generates data automatically
python ab_test_analysis.py

# 4. View outputs
# → outputs/charts/*.png        (4 charts)
# → outputs/ab_report.txt       (full written report)
# → outputs/ab_experiment_data.csv
```

---

## Charts Generated

### 01 — Sample Size Curve
Shows how required sample size grows as the minimum detectable effect shrinks. Demonstrates why you must calculate n *before* running a test.

### 02 — A/B Test Results Dashboard
Four-panel: CTR comparison, 95% CI for the lift, session duration distribution, and conversion rate by device type.

### 03 — Peeking Bias Demonstration
Shows how p-value fluctuates throughout a test. Illustrates why stopping the moment you see p < 0.05 causes false positives — even when there is no real effect.

### 04 — Decision Report Card
Clean summary table of all results and the final ship/no-ship decision.

---

## 5 Common A/B Testing Mistakes

| Mistake | Why it's dangerous | Fix |
|---------|-------------------|-----|
| **Peeking** | Reading early inflates false positive rate | Pre-determine n, wait until complete |
| **Sample ratio mismatch** | Broken randomisation invalidates results | Check 50/50 split before reading |
| **Multiple testing** | 20 metrics × α=0.05 → 1 false positive by chance | Apply Bonferroni or FDR correction |
| **Contamination** | Users see both variants → groups not independent | Enforce strict user-level bucketing |
| **Underpowered test** | Misses real effects, wastes time | Always calculate n before starting |

---

## Business Recommendation

> Roll out the blue button.
>
> The 22% relative lift in CTR is statistically significant (p = 0.0024) and the 95% confidence interval is entirely positive ([+0.76pp, +3.52pp]), confirming the effect is not due to random chance.
>
> At 10,000 daily visitors, this represents approximately **214 additional conversions per day**.
