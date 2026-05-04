"""
====================================================================
  A/B Testing & Statistical Hypothesis Testing — Complete Framework
  Author  : Saiteja Chedadeepu
  GitHub  : https://github.com/Chedadeepu
  Dataset : Self-generated (no download required — just run!)
====================================================================
  Covers:
    ✓ Experiment design  ✓ Sample size calculation
    ✓ Z-test             ✓ Confidence intervals
    ✓ Cohen's d           ✓ Bonferroni correction
    ✓ Peeking bias demo  ✓ 5 common A/B pitfalls
====================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import NormalIndPower

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

# ── output folders ────────────────────────────────────────────────
CHART_DIR  = os.path.join("outputs", "charts")
OUTPUT_DIR = "outputs"
os.makedirs(CHART_DIR, exist_ok=True)

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════
# STEP 1 — EXPERIMENT DESIGN & SAMPLE SIZE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  STEP 1 — Experiment Design & Sample Size Calculation")
print("="*62)

BASELINE = 0.10   # current conversion rate  (Control)
TARGET   = 0.12   # expected rate after change (Treatment)
ALPHA    = 0.05   # significance level
POWER    = 0.80   # desired power

es = proportion_effectsize(BASELINE, TARGET)
n  = int(NormalIndPower().solve_power(
        effect_size=es, alpha=ALPHA, power=POWER,
        alternative="two-sided")) + 1

print(f"""
  Business context : Homepage button colour change
  Control (old)    : Grey button → CTR = {BASELINE:.0%}
  Treatment (new)  : Blue button → CTR = {TARGET:.0%} (expected)
  Min detectable   : {(TARGET-BASELINE)*100:.0f} percentage-point lift
  Significance (α) : {ALPHA}
  Power (1-β)      : {POWER:.0%}
  ──────────────────────────────────────────────────────────
  Required n/group : {n:,}
  Total users      : {n*2:,}
  → Do NOT read results until both groups reach {n:,} users!
""")

# sample-size vs effect-size chart
effects   = np.linspace(0.005, 0.09, 80)
ns        = []
for eff in effects:
    _es = proportion_effectsize(BASELINE, BASELINE + eff)
    _n  = int(NormalIndPower().solve_power(
            effect_size=_es, alpha=ALPHA, power=POWER,
            alternative="two-sided")) + 1
    ns.append(_n)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(effects * 100, ns, color="#2563EB", linewidth=2.5)
ax.fill_between(effects * 100, ns, alpha=0.08, color="#2563EB")
ax.axvline((TARGET - BASELINE) * 100, color="#EF4444",
           ls="--", lw=2, label=f"Our target effect (+{(TARGET-BASELINE)*100:.0f}pp)")
ax.axhline(n, color="#EF4444", ls=":", lw=2,
           label=f"Required n = {n:,}")
ax.set_xlabel("Minimum Detectable Effect (percentage points)")
ax.set_ylabel("Required Sample Size per Group")
ax.set_title("Sample Size vs Detectable Effect  (α=0.05, Power=80%)",
             fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend()
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/01_sample_size_curve.png", dpi=150)
plt.close()
print("  ✓ Chart saved: 01_sample_size_curve.png")

# ═══════════════════════════════════════════════════════════════════
# STEP 2 — SIMULATE EXPERIMENT DATA
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  STEP 2 — Simulating Experiment Data")
print("="*62)

ctrl_conversions = np.random.binomial(1, BASELINE, n)
trt_conversions  = np.random.binomial(1, TARGET,   n)

df = pd.DataFrame({
    "user_id"     : range(1, 2 * n + 1),
    "group"       : ["Control"] * n + ["Treatment"] * n,
    "converted"   : np.concatenate([ctrl_conversions, trt_conversions]),
    "session_sec" : np.concatenate([
        np.abs(np.random.normal(185, 55, n)),
        np.abs(np.random.normal(198, 55, n))
    ]),
    "pages_viewed": np.concatenate([
        np.random.poisson(3.2, n).clip(1, 20),
        np.random.poisson(3.6, n).clip(1, 20)
    ]),
    "device": np.random.choice(
        ["Desktop", "Mobile", "Tablet"],
        2 * n, p=[0.55, 0.35, 0.10]
    )
})

df.to_csv(f"{OUTPUT_DIR}/ab_experiment_data.csv", index=False)
print(f"""
  Users simulated : {len(df):,}  (Control: {n:,} | Treatment: {n:,})
  Dataset saved   : outputs/ab_experiment_data.csv
""")

# ═══════════════════════════════════════════════════════════════════
# STEP 3 — SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  STEP 3 — Summary Statistics")
print("="*62)

grp = df.groupby("group").agg(
    Users       = ("user_id",    "count"),
    Conversions = ("converted",  "sum"),
    CTR_pct     = ("converted",  lambda x: round(x.mean()*100, 3)),
    Avg_Session = ("session_sec","mean"),
    Avg_Pages   = ("pages_viewed","mean")
).reset_index()

print(f"\n{grp.to_string(index=False)}\n")

ctrl_ctr = df[df["group"] == "Control"]["converted"].mean()
trt_ctr  = df[df["group"] == "Treatment"]["converted"].mean()
lift_abs = trt_ctr - ctrl_ctr
lift_rel = lift_abs / ctrl_ctr * 100

print(f"  Absolute lift : {lift_abs*100:+.3f} percentage points")
print(f"  Relative lift : {lift_rel:+.2f}%")

# ═══════════════════════════════════════════════════════════════════
# STEP 4 — STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  STEP 4 — Hypothesis Tests")
print("="*62)

# 4a. Two-proportion z-test
ctrl_conv = int(df[df["group"] == "Control"]["converted"].sum())
trt_conv  = int(df[df["group"] == "Treatment"]["converted"].sum())
z_stat, p_val = proportions_ztest(
    [ctrl_conv, trt_conv], [n, n], alternative="two-sided")

# 95% CI for lift
z_crit  = norm.ppf(1 - ALPHA / 2)
se_lift = np.sqrt(ctrl_ctr*(1-ctrl_ctr)/n + trt_ctr*(1-trt_ctr)/n)
ci_lo   = lift_abs - z_crit * se_lift
ci_hi   = lift_abs + z_crit * se_lift

decision = ("REJECT H₀ ✓ — Ship the new button!"
            if p_val < ALPHA
            else "FAIL TO REJECT H₀ — Do not ship yet")

print(f"""
  ── Primary Test: Two-Proportions Z-Test ──────────────────
  H₀ : CTR_control = CTR_treatment (no difference)
  H₁ : CTR_control ≠ CTR_treatment (two-sided)

  Control CTR   : {ctrl_ctr:.4f}  ({ctrl_conv:,} conversions / {n:,} users)
  Treatment CTR : {trt_ctr:.4f}  ({trt_conv:,} conversions / {n:,} users)
  Absolute lift : {lift_abs*100:+.3f} pp
  Relative lift : {lift_rel:+.2f}%
  z-statistic   : {z_stat:.4f}
  p-value       : {p_val:.4f}
  95% CI (lift) : [{ci_lo*100:+.3f}pp,  {ci_hi*100:+.3f}pp]

  Decision      : {decision}
""")

# 4b. Cohen's d — practical significance
pooled_p = (ctrl_ctr + trt_ctr) / 2
cohens_d = lift_abs / np.sqrt(pooled_p * (1 - pooled_p))
eff_lbl  = ("negligible (<0.2)" if abs(cohens_d) < 0.2 else
            "small (0.2–0.5)"  if abs(cohens_d) < 0.5 else
            "medium (0.5–0.8)" if abs(cohens_d) < 0.8 else
            "large (>0.8)")

print(f"  ── Effect Size: Cohen's d ────────────────────────────────")
print(f"  d = {cohens_d:.4f}  →  {eff_lbl}")
print(f"  (Statistical significance ≠ practical significance!)\n")

# 4c. Secondary metrics — t-tests
ctrl_ses = df[df["group"] == "Control"]["session_sec"]
trt_ses  = df[df["group"] == "Treatment"]["session_sec"]
t_ses, p_ses = stats.ttest_ind(ctrl_ses, trt_ses)

ctrl_pgs = df[df["group"] == "Control"]["pages_viewed"]
trt_pgs  = df[df["group"] == "Treatment"]["pages_viewed"]
t_pgs, p_pgs = stats.ttest_ind(ctrl_pgs, trt_pgs)

print(f"  ── Secondary Metrics (t-tests) ───────────────────────────")
print(f"  Session Duration : ctrl={ctrl_ses.mean():.1f}s  trt={trt_ses.mean():.1f}s  "
      f"p={p_ses:.4f}  {'✓ sig' if p_ses < ALPHA else '✗ not sig'}")
print(f"  Pages Viewed     : ctrl={ctrl_pgs.mean():.2f}   trt={trt_pgs.mean():.2f}   "
      f"p={p_pgs:.4f}  {'✓ sig' if p_pgs < ALPHA else '✗ not sig'}")

# 4d. Bonferroni correction
metrics   = ["CTR", "Session Duration", "Pages Viewed"]
raw_p     = [p_val, p_ses, p_pgs]
bonf_alpha = ALPHA / len(metrics)

print(f"\n  ── Bonferroni Correction (3 metrics tested) ──────────────")
print(f"  Corrected α = {ALPHA} / {len(metrics)} = {bonf_alpha:.4f}")
for m, p in zip(metrics, raw_p):
    flag = "✓ Significant" if p < bonf_alpha else "✗ Not significant"
    print(f"  {m:<20}: p = {p:.4f}  →  {flag}")

# ═══════════════════════════════════════════════════════════════════
# STEP 5 — VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  STEP 5 — Charts")
print("="*62)

# --- Chart 2: Main results dashboard ---
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    f"A/B Test Results  |  p = {p_val:.4f}  |  Lift = {lift_rel:+.1f}%  |  {decision}",
    fontsize=12, fontweight="bold")

# 2a. CTR bar
bar_colors = ["#6B7280", "#2563EB"]
bars = axes[0,0].bar(
    ["Control\n(Grey Button)", "Treatment\n(Blue Button)"],
    [ctrl_ctr * 100, trt_ctr * 100],
    color=bar_colors, width=0.45, edgecolor="white", linewidth=1.5)
axes[0,0].bar_label(bars,
    labels=[f"{v:.2f}%" for v in [ctrl_ctr*100, trt_ctr*100]],
    padding=5, fontsize=13, fontweight="bold")
axes[0,0].set_ylabel("Conversion Rate (%)")
axes[0,0].set_title("Conversion Rate Comparison")
axes[0,0].set_ylim(0, max(ctrl_ctr, trt_ctr) * 100 * 1.3)
axes[0,0].axhline(ctrl_ctr * 100, color="#EF4444",
                  ls="--", lw=1.5, alpha=0.7, label="Baseline")
axes[0,0].legend(fontsize=10)

# 2b. 95% CI
x_mid = lift_abs * 100
x_err_lo = (lift_abs - ci_lo) * 100
x_err_hi = (ci_hi - lift_abs) * 100
axes[0,1].barh(
    ["Lift"], [x_mid],
    xerr=[[x_err_lo], [x_err_hi]],
    color="#2563EB", alpha=0.8, height=0.25, capsize=12,
    error_kw={"elinewidth": 2, "capthick": 2, "ecolor": "#1E3A8A"})
axes[0,1].axvline(0, color="black", lw=1.5, ls="--", alpha=0.6)
axes[0,1].set_xlabel("Lift (percentage points)")
axes[0,1].set_title(
    f"95% Confidence Interval\n[{ci_lo*100:+.2f}pp, {ci_hi*100:+.2f}pp]")
verdict_txt = ("✓ CI entirely above zero\n→ Effect is real"
               if ci_lo > 0 else
               "⚠ CI crosses zero\n→ Inconclusive, need more data")
verdict_color = "#22C55E" if ci_lo > 0 else "#EF4444"
axes[0,1].text(
    0.97, 0.80, verdict_txt, transform=axes[0,1].transAxes,
    ha="right", fontsize=10, color=verdict_color,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85))

# 2c. Session duration
for grp_name, color, ls in [("Control","#6B7280","solid"),
                              ("Treatment","#2563EB","solid")]:
    data = df[df["group"] == grp_name]["session_sec"]
    axes[1,0].hist(data, bins=40, alpha=0.55, color=color,
                   label=f"{grp_name} (μ={data.mean():.0f}s)",
                   edgecolor="white")
axes[1,0].set_xlabel("Session Duration (seconds)")
axes[1,0].set_ylabel("Users")
axes[1,0].set_title("Session Duration Distribution")
axes[1,0].legend()

# 2d. Conversion by device
dev_rate = (df.groupby(["group","device"])["converted"]
              .mean().mul(100).reset_index())
dev_pivot = dev_rate.pivot(
    index="device", columns="group", values="converted")
dev_pivot.plot(kind="bar", ax=axes[1,1],
               color=bar_colors, width=0.5,
               edgecolor="white", linewidth=1.5)
axes[1,1].set_ylabel("Conversion Rate (%)")
axes[1,1].set_title("Conversion Rate by Device")
axes[1,1].set_xlabel("")
axes[1,1].legend(title="Group")
plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/02_ab_test_results.png", dpi=150)
plt.close()
print("  ✓ Chart saved: 02_ab_test_results.png")

# --- Chart 3: Peeking bias ---
running_p_vals = []
check_points   = range(100, n + 1, 50)
for k in check_points:
    c_ = ctrl_conversions[:k].sum()
    t_ = trt_conversions[:k].sum()
    # avoid edge case where both sums are 0 or both n
    if 0 < c_ < k and 0 < t_ < k:
        _, p_ = proportions_ztest([c_, t_], [k, k])
    else:
        p_ = 1.0
    running_p_vals.append(p_)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(list(check_points), running_p_vals,
        color="#7C3AED", lw=2, label="p-value (checked at each point)")
ax.axhline(ALPHA, color="#EF4444", ls="--", lw=2,
           label=f"α = {ALPHA} threshold")
ax.axvline(n, color="#22C55E", ls="--", lw=2,
           label=f"Pre-planned n = {n:,} ← only valid read point")
ax.fill_between(
    list(check_points), running_p_vals, ALPHA,
    where=[p < ALPHA for p in running_p_vals],
    alpha=0.15, color="#EF4444", label="False-positive danger zone")
ax.set_xlabel("Users per Group (test running…)")
ax.set_ylabel("p-value")
ax.set_title(
    "Peeking Bias Demonstration\n"
    "→ p-value fluctuates! Reading early can cause false positives.",
    fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/03_peeking_bias.png", dpi=150)
plt.close()
print("  ✓ Chart saved: 03_peeking_bias.png")

# --- Chart 4: Sample size curve (already saved as 01) ---

# --- Chart 5: Summary infographic ---
fig, ax = plt.subplots(figsize=(10, 7))
ax.axis("off")
title = "A/B Test — Final Decision Report"
ax.text(0.5, 0.97, title, ha="center", va="top",
        fontsize=16, fontweight="bold", color="#1B3A6B",
        transform=ax.transAxes)

rows = [
    ("Experiment",        "Homepage Button Colour  (Grey → Blue)"),
    ("Users tested",      f"{len(df):,}  ({n:,} per group)"),
    ("Control CTR",       f"{ctrl_ctr:.2%}"),
    ("Treatment CTR",     f"{trt_ctr:.2%}"),
    ("Absolute lift",     f"{lift_abs*100:+.3f} pp"),
    ("Relative lift",     f"{lift_rel:+.2f}%"),
    ("z-statistic",       f"{z_stat:.4f}"),
    ("p-value",           f"{p_val:.4f}"),
    ("Significance (α)",  f"{ALPHA}"),
    ("95% CI (lift)",     f"[{ci_lo*100:+.2f}pp, {ci_hi*100:+.2f}pp]"),
    ("Cohen's d",         f"{cohens_d:.4f}  ({eff_lbl})"),
    ("Bonferroni α",      f"{bonf_alpha:.4f}  (3 metrics tested)"),
    ("DECISION",          decision),
]

for i, (label, value) in enumerate(rows):
    y_pos = 0.90 - i * 0.063
    bg    = "#EFF6FF" if i % 2 == 0 else "white"
    dec_r = i == len(rows) - 1
    ax.add_patch(plt.Rectangle(
        (0.0, y_pos - 0.025), 1.0, 0.055,
        transform=ax.transAxes, color=("#D1FAE5" if dec_r else bg),
        zorder=0))
    ax.text(0.03, y_pos, label, transform=ax.transAxes,
            fontsize=10.5, fontweight="bold",
            color="#1B3A6B" if dec_r else "#374151", va="center")
    ax.text(0.42, y_pos, value, transform=ax.transAxes,
            fontsize=10.5, color="#22C55E" if dec_r else "#111827",
            fontweight="bold" if dec_r else "normal", va="center")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/04_decision_report.png", dpi=150,
            bbox_inches="tight")
plt.close()
print("  ✓ Chart saved: 04_decision_report.png")

# ═══════════════════════════════════════════════════════════════════
# STEP 6 — WRITTEN REPORT
# ═══════════════════════════════════════════════════════════════════
report = f"""
======================================================================
  A/B TEST ANALYSIS REPORT
  Author    : Saiteja Chedadeepu  |  github.com/Chedadeepu
  Scenario  : Homepage button colour optimisation
======================================================================

EXPERIMENT DESIGN
  Control      : Grey button   →  baseline CTR = {BASELINE:.0%}
  Treatment    : Blue button   →  expected CTR = {TARGET:.0%}
  Alpha (α)    : {ALPHA}
  Power (1-β)  : {POWER:.0%}
  Required n   : {n:,} per group  ({n*2:,} total)

RESULTS
  Control CTR  : {ctrl_ctr:.4f}  ({ctrl_conv:,} / {n:,})
  Treatment CTR: {trt_ctr:.4f}  ({trt_conv:,} / {n:,})
  Absolute lift: {lift_abs*100:+.3f} pp
  Relative lift: {lift_rel:+.2f}%
  z-stat       : {z_stat:.4f}
  p-value      : {p_val:.4f}
  95% CI       : [{ci_lo*100:+.2f}pp, {ci_hi*100:+.2f}pp]
  Cohen's d    : {cohens_d:.4f}  ({eff_lbl})

DECISION
  {decision}

BONFERRONI CORRECTION  (3 metrics → corrected α = {bonf_alpha:.4f})
  CTR              : p={p_val:.4f}  {"✓ sig" if p_val<bonf_alpha else "✗ not sig"}
  Session Duration : p={p_ses:.4f}  {"✓ sig" if p_ses<bonf_alpha else "✗ not sig"}
  Pages Viewed     : p={p_pgs:.4f}  {"✓ sig" if p_pgs<bonf_alpha else "✗ not sig"}

5 COMMON A/B TESTING PITFALLS
  1. Peeking      — Stopping early inflates Type I error. Always
                    pre-calculate n and wait until you reach it.
  2. Sample ratio — If groups deviate from 50/50, randomisation
                    is broken. Check chi-square p > 0.05.
  3. Multiple     — Testing 20 metrics × α=0.05 → 1 false positive
     testing        expected by chance. Apply Bonferroni correction.
  4. Contamination— Users seeing both variants break independence.
                    Enforce strict user-level assignment.
  5. Under-powered— Running without sample size calc means you will
                    miss real effects. Always calculate n first.

BUSINESS RECOMMENDATION
  At 10,000 daily visitors, a {lift_rel:+.1f}% relative CTR lift =
  approximately {int(10000 * lift_abs):,} additional conversions per day.
  Roll out the blue button.

======================================================================
"""
with open("outputs/final_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
     
# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  ALL DONE — Key Results")
print("="*62)
print(f"""
  Control CTR   : {ctrl_ctr:.2%}
  Treatment CTR : {trt_ctr:.2%}
  Lift          : {lift_rel:+.1f}%
  p-value       : {p_val:.4f}
  95% CI        : [{ci_lo*100:+.2f}pp, {ci_hi*100:+.2f}pp]
  Cohen's d     : {cohens_d:.4f}  ({eff_lbl})

  Decision      : {decision}

  Outputs
  ─────────────────────────────────────────────────────────
  outputs/charts/01_sample_size_curve.png
  outputs/charts/02_ab_test_results.png
  outputs/charts/03_peeking_bias.png
  outputs/charts/04_decision_report.png
  outputs/ab_experiment_data.csv
  outputs/ab_report.txt
""")
