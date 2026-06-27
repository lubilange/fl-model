import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from supabase import create_client, Client

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Dashboard", layout="wide")

# =========================
# STYLE CSS
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
default_state = {
    "model_loaded": False,
    "trained": False,
    "metrics": {},
    "history": []
}

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================
# FUNCTIONS
# =========================
def safe_fetch(table_name):
    try:
        response = supabase.table(table_name).select("*").execute()
        return response.data or []
    except Exception as e:
        st.warning(f"Impossible de charger la table '{table_name}' : {e}")
        return []


def has_column(df, column_name):
    return not df.empty and column_name in df.columns


def render_card(title, value):
    st.markdown(
        f"""
        <div class="card">
            <h3>{title}</h3>
            <h2>{value}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# LOAD DATA
# =========================
patients = pd.DataFrame(safe_fetch("patients"))
conditions = pd.DataFrame(safe_fetch("conditions"))
observations = pd.DataFrame(safe_fetch("observations"))
treatments = pd.DataFrame(safe_fetch("treatments"))
adherence_logs = pd.DataFrame(safe_fetch("adherence_logs"))
nurses = pd.DataFrame(safe_fetch("nurses"))

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

# =========================================================
# DASHBOARD CLINIQUE
# =========================================================
if menu == "Dashboard Clinique":
    st.subheader("🏥 Vue clinique en temps réel")

    col1, col2, col3 = st.columns(3)

    with col1:
        render_card("👥 Patients", len(patients))

    with col2:
        render_card("🧾 Conditions FHIR", len(conditions))

    with col3:
        render_card("🩺 Symptômes", len(observations))

    st.divider()

    st.markdown("### 🚨 Niveau d'alerte")
    if has_column(conditions, "severity"):
        st.bar_chart(conditions["severity"].value_counts())
    else:
        st.info("Aucune donnée de sévérité disponible pour les conditions.")

    st.divider()

    st.markdown("### 📈 Répartition des symptômes")
    if has_column(observations, "severity"):
        symptom_dist = observations["severity"].value_counts()
        st.bar_chart(symptom_dist)
    else:
        st.info("Aucune donnée de sévérité disponible pour les symptômes.")

    st.divider()

    st.markdown("### 👩‍⚕️ Support infirmier")
    if has_column(nurses, "status"):
        st.bar_chart(nurses["status"].value_counts())
    else:
        st.info("Aucun statut infirmier disponible.")

    st.divider()

    st.markdown("### Cas de simulations")

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
# DASHBOARD RECHERCHE
# =========================================================
elif menu == "Dashboard Recherche":
    st.subheader("📈 Graphique pour Recherche Analytique")

    st.markdown("### Répartition patients")

    if has_column(patients, "gender"):
        gender_dist = patients["gender"].value_counts()
        st.bar_chart(gender_dist)
        st.write("Distribution des patients par genre.")
    else:
        st.info("Aucune donnée de genre disponible.")

    st.markdown("### Répartition des risques")

    if has_column(conditions, "severity"):
        risk_dist = conditions["severity"].value_counts()

        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=risk_dist.index,
            values=risk_dist.values,
            hole=0.3
        ))

        fig.update_layout(
            title="Répartition des risques par sévérité"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune donnée de risque disponible.")

# =========================================================
# EXPORT ANONYMISÉ
# =========================================================
elif menu == "Export Anonymisé":
    st.subheader("🔬 Export anonymisé pour analyses")

    data = []

    data.append({
        "Catégorie": "Patients",
        "Indicateur": "Nombre total",
        "Valeur": len(patients)
    })

    if has_column(patients, "gender"):
        for genre, nb in patients["gender"].value_counts().items():
            data.append({
                "Catégorie": "Patients",
                "Indicateur": f"Genre {genre}",
                "Valeur": nb
            })

    data.append({
        "Catégorie": "Observations",
        "Indicateur": "Nombre total",
        "Valeur": len(observations)
    })

    if has_column(observations, "severity"):
        for sev, nb in observations["severity"].value_counts().items():
            data.append({
                "Catégorie": "Observations",
                "Indicateur": f"Sévérité {sev}",
                "Valeur": nb
            })

    data.append({
        "Catégorie": "Conditions",
        "Indicateur": "Nombre total",
        "Valeur": len(conditions)
    })

    if has_column(conditions, "severity"):
        for sev, nb in conditions["severity"].value_counts().items():
            data.append({
                "Catégorie": "Conditions",
                "Indicateur": f"Sévérité {sev}",
                "Valeur": nb
            })

    data.append({
        "Catégorie": "Infirmiers",
        "Indicateur": "Nombre total",
        "Valeur": len(nurses)
    })

    if has_column(nurses, "status"):
        for stat, nb in nurses["status"].value_counts().items():
            data.append({
                "Catégorie": "Infirmiers",
                "Indicateur": f"Statut {stat}",
                "Valeur": nb
            })

    data.append({
        "Catégorie": "Adhésion",
        "Indicateur": "Nombre total de logs",
        "Valeur": len(adherence_logs)
    })

    if has_column(adherence_logs, "status"):
        for stat, nb in adherence_logs["status"].value_counts().items():
            data.append({
                "Catégorie": "Adhésion",
                "Indicateur": f"Statut {stat}",
                "Valeur": nb
            })

    data.append({
        "Catégorie": "Traitements",
        "Indicateur": "Nombre total",
        "Valeur": len(treatments)
    })

    df_export = pd.DataFrame(data)

    st.dataframe(df_export, use_container_width=True)

    csv = df_export.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Télécharger l'export anonymisé CSV",
        data=csv,
        file_name="export_anonymise.csv",
        mime="text/csv"
    )
