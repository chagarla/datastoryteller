import os, io, json, textwrap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from openai import OpenAI

# -------- Config --------
st.set_page_config(page_title="Data Storyteller", page_icon="📊", layout="centered")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
DEFAULT_MODEL = "gpt-4o-mini"  # cheap + solid

st.title("📊 Data Storyteller")
st.write("Upload a CSV or generate sample data. I’ll chart it and narrate the story.")

# -------- Sidebar: Controls & Costs (NEW) --------
with st.sidebar:
    st.markdown("### Controls")
    use_sample = st.checkbox("Generate sample dataset", value=True)
    use_llm = st.checkbox("Use LLM narration (costs tokens)", value=False)
    max_tokens = st.slider("Max tokens (response cap)", 150, 800, 300, help="Lower = cheaper")
    st.caption("Tip: keep tokens small for short, punchy exec summaries.")

    st.markdown("### Cost guard (rough estimate)")
    input_tok_est = st.slider("Input tokens estimate", 400, 2400, 900, step=100,
                              help="Very rough; ~4 chars ≈ 1 token.")
    # NOTE: Update prices if/when they change. This is a sanity check, not a bill.
    per_1k_in = 0.0_1  # $0.01 per 1K input tokens (placeholder; check your pricing)
    per_1k_out = 0.0_3 # $0.03 per 1K output tokens (placeholder)
    est_cost = (input_tok_est/1000)*per_1k_in + (max_tokens/1000)*per_1k_out
    st.write(f"**Estimated cost/run:** ~${est_cost:0.4f}")
    st.caption("This is an estimate. Actual billing depends on exact tokens.")

# -------- Data ingest or generation (UPDATED) --------
file = None
df = None

if use_sample:
    st.subheader("Sample data generator")
    n_points = st.slider("Length (rows)", 50, 500, 180, step=10)
    noise = st.slider("Noise level", 0.0, 2.0, 0.4, 0.1)
    anomaly_count = st.slider("Anomalies", 0, 10, 3)
    metric_name = st.text_input("Metric name", "completions")

    # create a date index (daily)
    start = datetime.today() - timedelta(days=n_points)
    dates = pd.date_range(start=start, periods=n_points, freq="D")
    x = np.linspace(0, 3*np.pi, n_points)
    vals = 100 + 20*np.sin(x) + np.random.normal(0, noise*10, size=n_points)

    # inject anomalies
    if anomaly_count > 0:
        idxs = np.random.choice(np.arange(n_points), size=anomaly_count, replace=False)
        vals[idxs] += np.random.choice([40, -35, 60, -50], size=anomaly_count)

    df = pd.DataFrame({"date": dates, metric_name: vals})
    st.dataframe(df.tail(8))
    dt_col = "date"
    val_col = metric_name
else:
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        raw = file.read()
        df = pd.read_csv(io.BytesIO(raw))
        # guess datetime + numeric
        dt_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        num_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
        dt_col = st.selectbox("Pick datetime column", options=dt_candidates or df.columns.tolist())
        val_col = st.selectbox("Pick value column", options=num_candidates or df.columns.tolist())
    else:
        st.info("Upload a CSV or switch on 'Generate sample dataset' in the sidebar.")
        st.stop()

# -------- Prep & resample --------
try:
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
except Exception:
    st.error("Could not parse your datetime column.")
    st.stop()

df = df[[dt_col, val_col]].dropna().sort_values(dt_col).reset_index(drop=True)

freq = st.selectbox("Resample frequency", ["auto", "D (daily)", "H (hourly)", "W (weekly)"], index=0)
if freq != "auto":
    rule = freq.split(" ")[0]
    df = df.set_index(dt_col).resample(rule)[val_col].sum().reset_index()
else:
    if (df[dt_col].diff().dt.total_seconds().median() or 0) > 36_000:
        df = df.set_index(dt_col).resample("D")[val_col].sum().reset_index()

# -------- Analysis --------
window = st.slider("Rolling window (baseline)", 3, 30, 7)
s = df[val_col].astype(float)
roll = s.rolling(window=window, min_periods=max(1, window//2)).mean()
std = s.rolling(window=window, min_periods=max(1, window//2)).std()
z = (s - roll) / (std.replace(0, np.nan))
z_threshold = st.slider("Anomaly z-threshold", 1.5, 4.0, 2.5, 0.1)
anomalies_idx = z.index[(z.abs() >= z_threshold) & (~z.isna())].tolist()
anomalies = df.loc[anomalies_idx, [dt_col, val_col]]

# trend
x = np.arange(len(df))
slope = np.polyfit(x, s.fillna(method="ffill").fillna(method="bfill")), 1)[0] if len(df) > 1 else 0.0
trend = "uptrend 📈" if slope > 0 else ("downtrend 📉" if slope < 0 else "flat ➖")

# peaks & troughs
top_n = min(3, len(df))
peaks = df.nlargest(top_n, val_col)[[dt_col, val_col]]
troughs = df.nsmallest(top_n, val_col)[[dt_col, val_col]]

# -------- Chart --------
st.subheader("Time Series")
fig = plt.figure()
plt.plot(df[dt_col], df[val_col], label=val_col)
if not roll.isna().all():
    plt.plot(df[dt_col], roll, label=f"Rolling mean ({window})")
if not anomalies.empty:
    plt.scatter(anomalies[dt_col], anomalies[val_col], label="Anomalies", marker="o")
plt.xlabel(dt_col)
plt.ylabel(val_col)
plt.legend()
st.pyplot(fig)

# -------- Narrative --------
def rule_based_story(ctx):
    lines = []
    lines.append(f"{ctx['metric']} from {ctx['start']} to {ctx['end']} shows a {ctx['trend']}.")
    if ctx["max"]:
        lines.append(f"Peak around {ctx['max']['t']} (~{ctx['max']['v']:.1f}); "
                     f"low near {ctx['min']['t']} (~{ctx['min']['v']:.1f}).")
    lines.append(f"Average ~{ctx['mean']:.1f}; σ ~{ctx['std']:.1f}.")
    if ctx["anomalies"]:
        lines.append(f"{len(ctx['anomalies'])} anomaly/anomalies flagged (z ≥ {ctx['z_threshold']}). "
                     "Consider checking deployments, traffic shifts, or upstream dependencies around those timestamps.")
    else:
        lines.append("No strong anomalies detected versus rolling baseline.")
    lines.append("Next steps: validate known events (releases/outages/promos) and monitor for persistence.")
    return " ".join(lines)

context = {
    "metric": val_col,
    "points": len(df),
    "start": str(df[dt_col].iloc[0]),
    "end": str(df[dt_col].iloc[-1]),
    "trend": trend,
    "slope": float(slope),
    "mean": float(s.mean()),
    "std": float(s.std()),
    "min": {"t": str(troughs[dt_col].iloc[0]), "v": float(troughs[val_col].iloc[0])} if len(troughs) else None,
    "max": {"t": str(peaks[dt_col].iloc[0]), "v": float(peaks[val_col].iloc[0])} if len(peaks) else None,
    "anomalies": [{"t": str(r[dt_col]), "v": float(r[val_col])} for _, r in anomalies.iterrows()][:10],
    "window": window,
    "z_threshold": z_threshold,
}

st.subheader("Narrative")
if not use_llm:
    st.write(rule_based_story(context))
else:
    if not client:
        st.error("OPENAI_API_KEY not set. Turn off 'Use LLM narration' or add your key as a secret.")
    else:
        with st.spinner("Writing the story…"):
            sys = "You are a precise analytics narrator. Be concise, non-repetitive, and executive-friendly."
            usr = f"""
Turn this JSON context into a brief narrative (4–7 sentences) explaining trend,
peaks/troughs, and why it matters. If anomalies exist, explain them and suggest 1–2 follow-ups.
Do not fabricate incidents; hedge when unsure.

Context JSON:
{json.dumps(context)}
"""
            try:
                resp = client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
                    temperature=0.2,
                    max_tokens=max_tokens
                )
                story = resp.choices[0].message.content.strip()
            except Exception as e:
                story = f"(LLM call failed) {rule_based_story(context)}"
        st.write(story)

# -------- Export --------
if st.button("Export narrative as TXT"):
    txt = f"Metric: {val_col}\nRange: {context['start']} → {context['end']}\n\n{rule_based_story(context) if not use_llm else story}\n"
    st.download_button("Download", data=txt, file_name="data_story.txt", mime="text/plain")
