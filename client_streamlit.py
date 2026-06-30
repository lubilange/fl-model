import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from supabase import create_client, Client

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Dashboard", layout="wide")

# =========================
# STYLE
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

section[data-testid="stSidebar"] input {
    color: black !important;
    background-color: white !important;
    caret-color: black !important;
}

section[data-testid="stSidebar"] input::placeholder {
    color: gray !important;
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
# FETCH SAFE
# =========================
def safe_fetch(table):
    try:
        return supabase.table(table).select("*").execute().data or []
    except Exception as e:
        st.warning(f"Erreur de chargement table {table}: {e}")
        return []

# =========================
# LOGIN ADMIN
# =========================
st.sidebar.markdown("## Connexion admin")

admin_email = st.sidebar.text_input("Email admin")
admin_password = st.sidebar.text_input("Mot de passe", type="password")

if not admin_email or not admin_password:
    st.warning("Veuillez entrer vos identifiants admin.")
    st.stop()

try:
    admin_user = (
        supabase.table("admins")
        .select("*, countries(name, code), provinces(name)")
        .eq("email", admin_email)
        .eq("password", admin_password)
        .maybe_single()
        .execute()
        .data
    )
except Exception as e:
    st.error(f"Erreur de connexion admin : {e}")
    st.stop()

if not admin_user:
    st.error("Accès refusé : admin non reconnu.")
    st.stop()

admin_country = admin_user.get("countries") or {}
admin_province = admin_user.get("provinces") or {}

admin_country_code = admin_country.get("code")
admin_country_name = admin_country.get("name")
admin_province_name = admin_province.get("name")

st.sidebar.success(f"Connecté : {admin_user.get('name')}")
st.sidebar.write(f"Pays : {admin_country_name}")

if admin_country_code == "RDC":
    st.sidebar.write(f"Province : {admin_province_name}")
else:
    st.sidebar.write("Zone : Hors RDC")

# =========================
# MENU
# =========================
menu = st.sidebar.radio(
    "Menu",
    [
        "Dashboard Clinique",
        "Dashboard Recherche",
        "Export Anonymisé"
    ]
)

# =========================
# CHARGEMENT DES TABLES
# =========================
patients = pd.DataFrame(safe_fetch("patients"))
conditions = pd.DataFrame(safe_fetch("conditions"))
observations = pd.DataFrame(safe_fetch("observations"))
treatments = pd.DataFrame(safe_fetch("treatments"))
adherence_logs = pd.DataFrame(safe_fetch("adherence_logs"))
nurses = pd.DataFrame(safe_fetch("nurses"))

# =========================
# FILTRE ADMIN
# =========================
def filter_by_admin_location(df, table_name):
    if df.empty:
        return df

    required_cols = {"country", "province"}

    if not required_cols.issubset(df.columns):
        st.error(f"Erreur : la table {table_name} doit avoir les colonnes country et province.")
        st.write("Colonnes disponibles :", df.columns.tolist())
        st.stop()

    if admin_country_code == "RDC":
        return df[
            (df["country"] == "RDC") &
            (df["province"] == admin_province_name)
        ]

    return df[
        (df["country"] != "RDC") |
        (df["province"].isna())
    ]

patients = filter_by_admin_location(patients, "patients")
nurses = filter_by_admin_location(nurses, "nurses")

patient_ids = patients["id"].tolist() if not patients.empty and "id" in patients.columns else []

if not observations.empty and "patient_id" in observations.columns:
    observations = observations[observations["patient_id"].isin(patient_ids)]

if not conditions.empty and "patient_id" in conditions.columns:
    conditions = conditions[conditions["patient_id"].isin(patient_ids)]

if not treatments.empty and "patient_id" in treatments.columns:
    treatments = treatments[treatments["patient_id"].isin(patient_ids)]

if not adherence_logs.empty and "patient_id" in adherence_logs.columns:
    adherence_logs = adherence_logs[adherence_logs["patient_id"].isin(patient_ids)]

# =========================
# DASHBOARD CLINIQUE
# =========================
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
        st.info("Aucune donnée de sévérité disponible pour cette zone.")

    st.divider()

    st.markdown("### 📈 Répartition des symptômes")
    if not observations.empty and "severity" in observations.columns:
        trend = observations["severity"].value_counts().reset_index()
        trend.columns = ["severity", "count"]
        st.bar_chart(trend.set_index("severity"))
    else:
        st.info("Aucune donnée de symptômes disponible pour cette zone.")

    st.divider()

    st.markdown("### 👩‍⚕️ Support infirmier")
    if not nurses.empty and "status" in nurses.columns:
        st.bar_chart(nurses["status"].value_counts())
    else:
        st.info("Aucun statut infirmier disponible pour cette zone.")

# =========================
# DASHBOARD RECHERCHE
# =========================
elif menu == "Dashboard Recherche":
    st.subheader("Graphique pour Recherche Analytique")

    st.markdown("### Répartition patients par genre")
    if not patients.empty and "gender" in patients.columns:
        gender_dist = patients["gender"].value_counts()
        st.bar_chart(gender_dist)
    else:
        st.info("Aucune donnée patient disponible pour cette zone.")

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
        st.info("Aucune condition disponible pour cette zone.")

# =========================
# EXPORT ANONYMISÉ
# =========================
elif menu == "Export Anonymisé":
    st.subheader("Export anonymisé pour analyses")

    if patients.empty:
        st.info("Aucun patient disponible pour cette zone.")
        st.stop()

    export_data = []

    for _, patient in patients.iterrows():
        patient_id = patient["id"]

        patient_conditions = conditions[
            conditions["patient_id"] == patient_id
        ] if not conditions.empty and "patient_id" in conditions.columns else pd.DataFrame()

        patient_observations = observations[
            observations["patient_id"] == patient_id
        ] if not observations.empty and "patient_id" in observations.columns else pd.DataFrame()

        patient_treatments = treatments[
            treatments["patient_id"] == patient_id
        ] if not treatments.empty and "patient_id" in treatments.columns else pd.DataFrame()

        export_data.append({
            "patient_id": patient_id,
            "gender": patient.get("gender"),
            "birth_date": patient.get("birth_date"),
            "country": patient.get("country"),
            "province": patient.get("province"),

            "nb_conditions": len(patient_conditions),
            "condition_severity":
                patient_conditions.iloc[-1]["severity"]
                if not patient_conditions.empty and "severity" in patient_conditions.columns
                else None,

            "nb_observations": len(patient_observations),
            "observation_text":
                " | ".join(patient_observations["text"].astype(str).tolist())
                if not patient_observations.empty and "text" in patient_observations.columns
                else None,

            "observation_severity":
                patient_observations.iloc[-1]["severity"]
                if not patient_observations.empty and "severity" in patient_observations.columns
                else None,

            "nb_treatments": len(patient_treatments),
            "treatments":
                " | ".join(patient_treatments["text"].astype(str).tolist())
                if not patient_treatments.empty and "text" in patient_treatments.columns
                else None
        })

    df_export = pd.DataFrame(export_data)

    st.dataframe(df_export, use_container_width=True)

    csv = df_export.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Télécharger l'export anonymisé (CSV)",
        csv,
        "export_anonymise_localisation.csv",
        "text/csv"
    )
