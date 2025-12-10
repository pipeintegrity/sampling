# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import beta, binom, norm, t, nct
from scipy.integrate import quad
from scipy.optimize import brentq

st.set_page_config(page_title="Statistical Calculator Suite", layout="wide")
st.title("Statistical Calculator Suite")
st.markdown("**Binomial CIs • Continuous CIs • OC Curves • Tolerance Limits** – All in your browser!")

# ------------------------------------------------------------------
# 1. Exact two-sided tolerance factor (kept from your code)
# ------------------------------------------------------------------
def k_two_sided_exact(n, P, gamma, epsabs=1e-10, epsrel=1e-8):
    from scipy.stats import chi2, ncx2
    alpha = 1.0 - gamma
    df = n - 1

    def integrand(z, k):
        q_P = ncx2.ppf(P, 1,z*z)
        t = df * q_P / (k*k)
        tail = 1.0 - chi2.cdf(t, df)
        return np.sqrt(2.0*n/np.pi) * np.exp(-0.5*n*z*z) * tail

    def F(k):
        val, _ = quad(lambda z: integrand(z,k), 0.0, np.inf,
                      epsabs=epsabs, epsrel=epsrel, limit=200)
        return val - (1.0-alpha)

    lo, hi = 1e-9, 10.0
    while F(hi) <= 0:
        hi *= 2
        if hi > 1e6:
            raise RuntimeError("Failed to bracket root")
    return brentq(F, lo, hi, xtol=1e-10)

# ------------------------------------------------------------------
# Sidebar – choose tool
# ------------------------------------------------------------------
tool = st.sidebar.selectbox(
    "Choose a tool",
    ["Binomial Confidence Intervals",
     "Continuous (Normal) Confidence Interval",
     "Operating Characteristic (OC) Curve",
     "Tolerance Limits"]
)

# ==================================================================
# 1. BINOMIAL CONFIDENCE INTERVALS
# ==================================================================
if tool == "Binomial Confidence Intervals":
    st.header("Binomial Confidence Intervals (Frequentist + Bayesian)")

    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Number of trials (n)", 1, 10000, 100)
        x = st.number_input("Number of successes (x)", 0, n, 75)
        conf = st.slider("Confidence / Credible level (%)", 50.0, 99.9, 95.0)
    with col2:
        method = st.selectbox("Frequentist method", [
            "Wilson", "Exact (Clopper-Pearson)",
            "Normal Approximation", "Agresti-Coull"
        ])
        prior_a = st.number_input("Bayesian prior α (Jeffreys = 0.5)", 0.01, 20.0, 0.5)
        prior_b = st.number_input("Bayesian prior β (Jeffreys = 0.5)", 0.01, 20.0, 0.5)

    alpha = 1 - conf/100

    # Frequentist CIs
    if method == "Wilson":
        z = stats.norm.ppf(1-alpha/2)
        p̂ = x/n
        denom = 1 + z**2/n
        center = (p̂ + z**2/(2*n)) / denom
        margin = z * np.sqrt((p̂*(1-p̂) + z**2/(4*n))/n) / denom
        freq_l, freq_u = center - margin, center + margin

    elif method == "Exact (Clopper-Pearson)":
        freq_l = beta.ppf(alpha/2, x, n-x+1) if x > 0 else 0.0
        freq_u = beta.ppf(1-alpha/2, x+1, n-x) if x < n else 1.0

    elif method == "Normal Approximation":
        p̂ = x/n
        se = np.sqrt(p̂*(1-p̂)/n)
        z = stats.norm.ppf(1-alpha/2)
        freq_l = max(0, p̂ - z*se)
        freq_u = min(1, p̂ + z*se)

    else:  # Agresti-Coull
        z = stats.norm.ppf(1-alpha/2)
        ñ = n + z**2
        p̃ = (x + z**2/2) / ñ
        se = np.sqrt(p̃*(1-p̃)/ñ)
        freq_l = max(0, p̃ - z*se)
        freq_u = min(1, p̃ + z*se)

    # Bayesian credible interval
    post_a = x + prior_a
    post_b = n - x + prior_b
    bayes_l = beta.ppf(alpha/2, post_a, post_b)
    bayes_u = beta.ppf(1-alpha/2, post_a, post_b)

    # Results
    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sample proportion p̂", f"{x/n:.4f}")
        st.write(f"**{method} {conf:.1f}% CI**")
        st.success(f"[{freq_l:.4f}, {freq_u:.4f}]")
    with col2:
        st.write(f"**Bayesian {conf:.1f}% Credible Interval** (Beta({prior_a}, {prior_b}))")
        st.success(f"[{bayes_l:.4f}, {bayes_u:.4f}]")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    # Frequentist
    ax1.hlines(1, freq_l, freq_u, color="steelblue", lw=5, label=f"{method}")
    ax1.plot([freq_l, freq_l], [0.9,1.1], color="steelblue")
    ax1.plot([freq_u, freq_u], [0.9,1.1], color="steelblue")
    ax1.plot(x/n, 1, "ro", markersize=8, label="p̂")
    ax1.set_xlim(max(0, freq_l-0.1), min(1, freq_u+0.1))
    ax1.set_yticks([])
    ax1.set_title(f"{method} Confidence Interval")
    ax1.legend()

    # Bayesian posterior
    xs = np.linspace(0, 1, 1000)
    ax2.plot(xs, beta.pdf(xs, post_a, post_b), color="forestgreen", lw=2)
    ax2.fill_between(xs, beta.pdf(xs, post_a, post_b), where=(xs>=bayes_l)&(xs<=bayes_u),
                     color="lightgreen", alpha=0.6, label=f"{conf:.1f}% Credible")
    ax2.axvline(x/n, color="red", ls="--", label="p̂")
    ax2.set_title(f"Bayesian Posterior (α={prior_a}, β={prior_b})")
    ax2.legend()

    st.pyplot(fig)

# ==================================================================
# 2. CONTINUOUS CONFIDENCE INTERVAL
# ==================================================================
elif tool == "Continuous (Normal) Confidence Interval":
    st.header("Confidence Interval for a Normal Mean (unknown σ)")

    col1, col2 = st.columns(2)
    with col1:
        mean = st.number_input("Sample mean (x̄)", value=50.0)
        s = st.number_input("Sample std dev (s)", min_value=0.01, value=10.0)
    with col2:
        n = st.number_input("Sample size (n)", 2, 10000, 30)
        conf = st.slider("Confidence level (%)", 50.0, 99.9, 95.0, key="cont")

    sem = s / np.sqrt(n)
    df = n - 1
    t_val = stats.t.ppf(1 - (1-conf/100)/2, df)
    margin = t_val * sem
    lower, upper = mean - margin, mean + margin

    st.metric("Margin of Error", f"±{margin:.4f}")
    st.success(f"**{conf:.1f}% CI for μ**:  [{lower:.4f}, {upper:.4f}]")

    fig, ax = plt.subplots(figsize=(8,4))
    xs = np.linspace(mean-4*s, mean+4*s, 1000)
    ax.plot(xs, stats.t.pdf(xs, df, mean, sem), color="steelblue")
    ax.fill_between(xs, stats.t.pdf(xs, df, mean, sem),
                    where=(xs>=lower)&(xs<=upper), color="lightblue", alpha=0.7)
    ax.axvline(mean, color="red", ls="--", label="Sample mean")
    ax.set_title(f"t-distribution (df={df})")
    ax.legend()
    st.pyplot(fig)

# ==================================================================
# 3. OC CURVE
# ==================================================================
elif tool == "Operating Characteristic (OC) Curve":
    st.header("Operating Characteristic (OC) Curve – Sampling Plan (n, c)")

    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Sample size n", 1, 1000, 50, key="oc_n")
        c = st.number_input("Acceptance number c", 0, n-1, 2, key="oc_c")
    with col2:
        rql = st.number_input("Rejectable Quality Level (p)", 0.0, 1.0, 0.10, step=0.01, key="oc_rql")

    p = np.linspace(0, 1, 1000)
    Pa = binom.cdf(c, n, p)

    # Find where Pa drops below 0.005 to set nice x-axis
    cutoff_idx = np.where(Pa < 0.005)[0]
    xmax = 1.0 if len(cutoff_idx)==0 else p[cutoff_idx[0]]*1.1

    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(p, Pa, label="OC Curve", lw=3, color="#2b7bba")
    pa_rql = binom.cdf(c, n, rql)
    ax.plot(rql, pa_rql, "ro")
    ax.axvline(rql, color="red", ls="--")
    ax.axhline(pa_rql, color="red", ls="--")
    ax.text(rql+0.02, pa_rql+0.05, f"Consumer's Risk = {pa_rql:.4f}",
            bbox=dict(facecolor="white", alpha=0.8))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("True defect rate p")
    ax.set_ylabel("Probability of Acceptance Pa")
    ax.set_title(f"OC Curve (n={n}, c={c})")
    ax.grid(True, alpha=0.4)
    st.pyplot(fig)

    st.info(f"**Consumer's Risk at p = {rql:.2%}**: {pa_rql:.4f}")

# ==================================================================
# 4. TOLERANCE LIMITS
# ==================================================================
else:  # Tolerance Limits
    st.header("Normal Tolerance Limits")

    col1, col2 = st.columns(2)
    with col1:
        mean = st.number_input("Sample mean", value=50.0, key="tol_mean")
        s = st.number_input("Sample std dev", min_value=0.01, value=10.0, key="tol_s")
        n = st.number_input("Sample size", 2, 10000, 30, key="tol_n")
    with col2:
        P = st.slider("Coverage (% of population)", 50.0, 99.9, 95.0)/100
        gamma = st.slider("Confidence (%)", 50.0, 99.9,99.0)/100
        tail = st.radio("Tail type", ["Two-sided", "One-sided lower", "One-sided upper"])

    if tail == "Two-sided":
        k = k_two_sided_exact(n, P, gamma)
        lower = mean - k * s
        upper = mean + k * s
        st.success(f"**{P*100:.1f}%/{gamma*100:.1f}% Two-sided Tolerance Interval**\n\n"
                   f"[{lower:.4f}, {upper:.4f}]\n\nk = {k:.4f}")
    else:
        z_p = stats.norm.ppf(P)
        nc = np.sqrt(n) * z_p
        k = stats.nct.ppf(gamma, n-1, nc) / np.sqrt(n)
        if "lower" in tail:
            lower = mean - k * s
            st.success(f"**Lower {P*100:.1f}%/{gamma*100:.1f}% Tolerance Limit**\n\n"
                       f"x > {lower:.4f}   (k = {k:.4f})")
        else:
            upper = mean + k * s
            st.success(f"**Upper {P*100:.1f}%/{gamma*100:.1f}% Tolerance Limit**\n\n"
                       f"x < {upper:.4f}   (k = {k:.4f})")

    # Plot
    xs = np.linspace(mean-4*s, mean+4*s, 1000)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(xs, norm.pdf(xs, mean, s), color="steelblue", lw=2)
    if tail == "Two-sided":
        ax.fill_between(xs, norm.pdf(xs, mean, s), where=(xs>=lower)&(xs<=upper),
                       color="lightgreen", alpha=0.6, label=f"Tolerance Interval")
    elif "lower" in tail:
        ax.fill_between(xs, norm.pdf(xs, mean, s), where=xs>=lower,
                       color="lightgreen", alpha=0.6)
    else:
        ax.fill_between(xs, norm.pdf(xs, mean, s), where=xs<=upper,
                       color="lightgreen", alpha=0.6)
    ax.axvline(mean, color="red", ls="--", label="Sample mean")
    ax.set_title("Normal Distribution with Tolerance Limits")
    ax.legend()
    st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made by **Joel Anderson** Total Integrity Analytics")