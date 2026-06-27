import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from supabase import create_client, Client

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Dashboard", layout="wide")

# =========================
# 🎨 STYLE CSS (sidebar + cartes)
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .main {
        background-color: #eef1f5;
    }

    section[data-testid="stSidebar"] {
        background-color: #1e3c5a;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }

    h1, h2, h3 {
        color: #1e3c5a;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# SUPABASE
# =========================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================
# SESSION STATE
# =========================
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False
if "trained" not in st.session_state:
    st.session_state["trained"] = False
if "metrics" not in st.session_state:
    st.session_state["metrics"] = {}
if "history" not in st.session_state:
    st.session_state["history"] = []

# =========================
# MENU
# =========================
menu = st.sidebar.radio(
    " Menu",
    [
        "Dashboard Clinique",
        "Dashboard Recherche",
        "Export Anonymisé"
    ]
)

# =========================
# SUPABASE SAFE FETCH
# =========================
def safe_fetch(table):
    try:
        return supabase.table(table).select("*").execute().data or []
    except Exception as e:
        st.warning(f"Erreur de chargement table {table}: {e}")
        return []

patients = pd.DataFrame(safe_fetch("patients"))
conditions = pd.DataFrame(safe_fetch("conditions"))
observations = pd.DataFrame(safe_fetch("observations"))
treatments = pd.DataFrame(safe_fetch("treatments"))
adherence_logs = pd.DataFrame(safe_fetch("adherence_logs"))
nurses = pd.DataFrame(safe_fetch("nurses"))

# =========================================================
# 📊 DASHBOARD CLINIQUE
# =========================================================
if menu == "Dashboard Clinique":
    st.subheader("🏥 Vue clinique en temps réel")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f'<div class="card"><h3>👥 Patients</h3><h2>{len(patients)}</h2></div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f'<div class="card"><h3>🧾 Conditions FHIR</h3><h2>{len(conditions)}</h2></div>',
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f'<div class="card"><h3>🩺 Symptômes</h3><h2>{len(observations)}</h2></div>',
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown("### 🚨 Niveau d'alerte")
    if not conditions.empty and "severity" in conditions.columns:
        st.bar_chart(conditions["severity"].value_counts())
    else:
        st.info("Aucune donnée de sévérité disponible")

    st.divider()

    st.markdown("### 📈 Répartition des symptômes")
    if not observations.empty and "severity" in observations.columns:
        trend = observations["severity"].value_counts().reset_index()
        trend.columns = ["severity", "count"]
        st.bar_chart(trend.set_index("severity"))
    else:
        st.info("Aucune donnée de symptômes disponible")

    st.divider()

    st.markdown("### 👩‍⚕️ Support infirmier")
    if not nurses.empty and "status" in nurses.columns:
        st.bar_chart(nurses["status"].value_counts())
    else:
        st.info("Aucun statut infirmier disponible")

    st.divider()

    st.markdown("### cas de simulations")
    sim = pd.DataFrame([
        {"glycémie": 4.5, "niveau": "normal"},
        {"glycémie": 8.2, "niveau": "élevé"},
        {"glycémie": 6.8, "niveau": "modéré"}
    ])

    sim["prediction"] = sim["glycémie"].apply(
        lambda x: "élevé" if x > 7 else "normal"
    )

    st.dataframe(sim, use_container_width=True)

# =========================================================
# 📈 DASHBOARD RECHERCHE
# =========================================================
elif menu == "Dashboard Recherche":
    st.subheader("Graphique pour Recherche Analytique")

    st.markdown("### Répartition patients")
    if not patients.empty and "gender" in patients.columns:
        gender_dist = patients["gender"].value_counts()
        st.bar_chart(gender_dist)
        st.write("Distribution patients par genre")
    else:
        st.info("Aucune donnée patient disponible")

    st.markdown("### Répartition des risques")
    if not conditions.empty and "severity" in conditions.columns:
        risk_dist = conditions["severity"].value_counts()

        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=risk_dist.index,
            values=risk_dist.values
        ))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune condition disponible")

# =========================================================
# 🔬 EXPORT ANONYMISÉ
# =========================================================
# =========================================================
# 🔬 EXPORT ANONYMISÉ
# =========================================================
elif menu == "Export Anonymisé":
    st.subheader("Export anonymisé pour analyses")

    if patients.empty:
        st.info("Aucun patient disponible")
        st.stop()

    export_data = []

    for _, patient in patients.iterrows():

        patient_id = patient["id"]

        patient_conditions = conditions[
            conditions["patient_id"] == patient_id
        ] if not conditions.empty else pd.DataFrame()

        patient_observations = observations[
            observations["patient_id"] == patient_id
        ] if not observations.empty else pd.DataFrame()

        patient_treatments = treatments[
            treatments["patient_id"] == patient_id
        ] if not treatments.empty else pd.DataFrame()

        export_data.append({
            "patient_id": patient_id,
            "gender": patient.get("gender"),
            "birth_date": patient.get("birth_date"),
            "onboarded": patient.get("onboarded"),

            "nb_conditions": len(patient_conditions),
            "condition_severity":
                patient_conditions.iloc[-1]["severity"]
                if not patient_conditions.empty and "severity" in patient_conditions.columns
                else None,

            "nb_observations": len(patient_observations),
            "observation_severity":
                patient_observations.iloc[-1]["severity"]
                if not patient_observations.empty and "severity" in patient_observations.columns
                else None,

            "nb_treatments": len(patient_treatments)
        })

    df_export = pd.DataFrame(export_data)

    st.dataframe(df_export, use_container_width=True)

    csv = df_export.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Télécharger l'export anonymisé (CSV)",
        csv,
        "export_anonymise.csv",
        "text/csv"
    )
