import os, io, textwrap, json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import timedelta
from openai import OpenAI

# --- config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Data Storyteller", page_icon="📊", layout="centered")
st.title("📊 Data Storyteller")
st.write("Upload a CSV with a datetime column + a numeric column. I’ll chart it and narrate the story.")

# --- sidebar help ---
with st.sidebar:
    st.markdown("**CSV format tips**")
    st.markdown("""
- Include a timestamp column (e.g., `date`, `time`, `timestamp`)
- Include **one numeric** metric column (e.g., `completions`, `sales`, `errors`)
- Example headers: `date,completions`
    """)

# --- upload ---
file = st.file_uploader("Upload CSV", type=["csv"])
if file:
    raw = file.read()
    df = pd.read_csv(io.BytesIO(raw))

    # guess datetime + value columns
    dt_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    num_candidates = [c for c in df.select_dtypes(include=[np.number]).columns]

    dt_col = st.selectbox("Pick datetime column", options=dt_candidates or df.columns.tolist())
    val_col = st.selectbox("Pick value column", options=num_candidates or df.columns.tolist())

    # parse + clean
    try:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    except Exception:
        st.error("Could not parse your datetime column. Try a different one.")
        st.stop()

    df = df[[dt_col, val_col]].dropna()
    df = df.sort_values(dt_col).reset_index(drop=True)

    # resample if needed (daily granularity default)
    freq = st.selectbox("Resample frequency", ["auto", "D (daily)", "H (hourly)", "W (weekly)"], index=0)
    if freq != "auto":
        rule = freq.split(" ")[0]
        df = df.set_index(dt_col).resample(rule)[val_col].sum().reset_index()
    else:
        # if irregular, coerce to daily sum
        if (df[dt_col].diff().dt.total_seconds().median() or 0) > 36_000:
            df = df.set_index(dt_col).resample("D")[val_col].sum().reset_index()

    # analysis
    window = st.slider("Rolling window (for baseline)", 3, 30, 7)
    s = df[val_col].astype(float)
    roll = s.rolling(window=window, min_periods=max(1, window//2)).mean()
    std = s.rolling(window=window, min_periods=max(1, window//2)).std()
    z = (s - roll) / (std.replace(0, np.nan))

    # anomaly detection
    z_threshold = st.slider("Anomaly z-score threshold", 1.5, 4.0, 2.5, 0.1)
    anomalies_idx = z.index[(z.abs() >= z_threshold) & (~z.isna())].tolist()
    anomalies = df.loc[anomalies_idx, [dt_col, val_col]]

    # trend (simple linear slope)
    x = np.arange(len(df))
    slope = np.polyfit(x, s.fillna(method="ffill").fillna(method="bfill"), 1)[0]
    trend = "uptrend 📈" if slope > 0 else ("downtrend 📉" if slope < 0 else "flat ➖")

    # peaks & troughs
    top_n = min(3, len(df))
    peaks = df.nlargest(top_n, val_col)[[dt_col, val_col]]
    troughs = df.nsmallest(top_n, val_col)[[dt_col, val_col]]

    # chart
    st.subheader("Time Series")
    fig = plt.figure()          # (1) matplotlib only, (2) single plot, (3) no colors set
    plt.plot(df[dt_col], df[val_col], label=val_col)
    if not roll.isna().all():
        plt.plot(df[dt_col], roll, label=f"Rolling mean ({window})")
    if not anomalies.empty:
        plt.scatter(anomalies[dt_col], anomalies[val_col], label="Anomalies", marker="o")
    plt.xlabel(dt_col)
    plt.ylabel(val_col)
    plt.legend()
    st.pyplot(fig)

    # facts to feed the LLM
    context = {
        "metric": val_col,
        "points": len(df),
        "start": str(df[dt_col].iloc[0]),
        "end": str(df[dt_col].iloc[-1]),
        "trend": trend,
        "slope": float(slope),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": {"t": str(df[dt_col].iloc[troughs.index[0]]), "v": float(troughs[val_col].iloc[0])} if len(troughs) else None,
        "max": {"t": str(df[dt_col].iloc[peaks.index[0]]), "v": float(peaks[val_col].iloc[0])} if len(peaks) else None,
        "anomalies": [{"t": str(r[dt_col]), "v": float(r[val_col])} for _, r in anomalies.iterrows()][:10],
        "window": window,
        "z_threshold": z_threshold,
    }

    st.subheader("Narrative")
    with st.spinner("Writing the story…"):
        sys = "You are a precise analytics narrator. Be concise, non-repetitive, and executive-friendly."
        usr = f"""
        Turn this JSON context into a brief narrative (4–7 sentences) explaining the trend,
        notable peaks/troughs, and the likely reasons a stakeholder should care. If anomalies exist,
        explain them and suggest 1–2 follow-ups. Do not fabricate incidents; hedge when uncertain.

        Context JSON:
        {json.dumps(context)}
        """
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
                temperature=0.2
            )
            story = resp.choices[0].message.content.strip()
        except Exception as e:
            story = f"(LLM call failed) Basic read: Data shows {trend}. Mean={context['mean']:.2f}. "\
                    f"Top value ~ {context['max']}. {len(context['anomalies'])} anomaly/ies flagged."

    st.write(story)

    # export
    if st.button("Export narrative as TXT"):
        txt = f"Metric: {val_col}\nRange: {context['start']} → {context['end']}\n\n{story}\n"
        st.download_button("Download", data=txt, file_name="data_story.txt", mime="text/plain")
else:
    st.info("Upload a CSV to get started. Need a sample? Create a file with columns like: `date,completions`")
