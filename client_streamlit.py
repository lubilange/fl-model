import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="FederatedSmartHealth",
    page_icon="🏥",
    layout="wide"
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #eef4fb 0%, #f8fbff 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10375c 0%, #0b253f 100%);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] input {
    color: #111 !important;
    background-color: white !important;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #10375c;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 15px;
    color: #5c6b7a;
    margin-bottom: 25px;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(16, 55, 92, 0.10);
    border: 1px solid #e8eef5;
    text-align: center;
    min-height: 130px;
}

.card h4 {
    color: #637083;
    font-size: 14px;
    margin-bottom: 8px;
}

.card h2 {
    color: #10375c;
    font-size: 34px;
    font-weight: 800;
    margin: 0;
}

.section-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(16, 55, 92, 0.08);
    border: 1px solid #e8eef5;
    margin-bottom: 18px;
}

.badge {
    display: inline-block;
    padding: 8px 14px;
    background: #e8f2ff;
    color: #10375c;
    border-radius: 30px;
    font-weight: 600;
    margin-bottom: 10px;
}

div.stButton > button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    border: none !important;
}

div.stButton > button[kind="primary"] {
    background-color: #16a34a !important;
    color: white !important;
}

.logout-btn button {
    background-color: #dc2626 !important;
    color: white !important;
}

.logout-btn button:hover {
    background-color: #b91c1c !important;
    color: white !important;
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
# FUNCTIONS
# =========================
def safe_fetch(table):
    try:
        return supabase.table(table).select("*").execute().data or []
    except Exception as e:
        st.warning(f"Erreur de chargement table {table}: {e}")
        return []


def normalize_gender(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in ["f", "female", "femme", "woman"]:
        return "F"

    if value in ["h", "m", "male", "homme", "man"]:
        return "H"

    return None


def make_card(title, value, icon=""):
    st.markdown(
        f"""
        <div class="card">
            <h4>{icon} {title}</h4>
            <h2>{value}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


def empty_info(text):
    st.info(text)


def filter_by_admin_location(df, table_name):
    if df.empty:
        return df

    required_cols = {"country", "province"}

    if not required_cols.issubset(df.columns):
        st.error(f"La table {table_name} doit avoir les colonnes country et province.")
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


def pie_chart(df, names, title):
    if df.empty or names not in df.columns:
        empty_info("Aucune donnée disponible.")
        return

    data = df[names].dropna().value_counts().reset_index()
    data.columns = [names, "count"]

    fig = px.pie(
        data,
        names=names,
        values="count",
        hole=0.45,
        title=title
    )
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)


def bar_count(df, column, title):
    if df.empty or column not in df.columns:
        empty_info("Aucune donnée disponible.")
        return

    data = df[column].dropna().value_counts().reset_index()
    data.columns = [column, "count"]

    fig = px.bar(
        data,
        x=column,
        y="count",
        title=title,
        text="count"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)


# =========================
# LOGIN SESSION
# =========================
if "admin_user" not in st.session_state:
    st.session_state.admin_user = None

if st.session_state.admin_user is None:
    st.sidebar.markdown("## Connexion")

    admin_email = st.sidebar.text_input("Email")
    admin_password = st.sidebar.text_input("Mot de passe", type="password")

    if st.sidebar.button("Se connecter", type="primary", use_container_width=True):
        if not admin_email or not admin_password:
            st.sidebar.error("Veuillez entrer vos identifiants.")
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
            st.sidebar.error(f"Erreur de connexion : {e}")
            st.stop()

        if not admin_user:
            st.sidebar.error("Identifiants incorrects.")
            st.stop()

        st.session_state.admin_user = admin_user
        st.rerun()

    st.markdown(
        """
        <div class="main-title">🏥 FederatedSmartHealth</div>
        <div class="subtitle">
        Plateforme de gestion de santé fédérée via WhatsApp, HL7/FHIR et apprentissage fédéré.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.warning("Veuillez vous connecter pour accéder au tableau de bord.")
    st.stop()

# =========================
# ADMIN CONNECTÉ
# =========================
admin_user = st.session_state.admin_user

admin_country = admin_user.get("countries") or {}
admin_province = admin_user.get("provinces") or {}

admin_country_code = admin_country.get("code")
admin_country_name = admin_country.get("name")
admin_province_name = admin_province.get("name")

st.sidebar.markdown("## Session")
st.sidebar.success(f"{admin_user.get('name')}")
st.sidebar.write(f"Pays : {admin_country_name}")

if admin_country_code == "RDC":
    st.sidebar.write(f"Province : {admin_province_name}")
    region_label = f"{admin_province_name} / RDC"
else:
    st.sidebar.write("Zone : Hors RDC")
    region_label = "Hors RDC"

st.sidebar.markdown('<div class="logout-btn">', unsafe_allow_html=True)
if st.sidebar.button("Se déconnecter", use_container_width=True):
    st.session_state.admin_user = None
    st.rerun()
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# =========================
# MENU
# =========================
menu = st.sidebar.radio(
    "Navigation",
    [
        "🏥 Clinique",
        "📲 WhatsApp",
        "🤖 Recherche & FL",
        "🔗 HL7 / FHIR",
        "👩‍⚕️ Infirmiers",
        "🛡️ Sécurité",
        "📤 Export Anonymisé"
    ]
)

# =========================
# DATA
# =========================
patients = pd.DataFrame(safe_fetch("patients"))
conditions = pd.DataFrame(safe_fetch("conditions"))
observations = pd.DataFrame(safe_fetch("observations"))
treatments = pd.DataFrame(safe_fetch("treatments"))
adherence_logs = pd.DataFrame(safe_fetch("adherence_logs"))
nurses = pd.DataFrame(safe_fetch("nurses"))

patients = filter_by_admin_location(patients, "patients")
nurses = filter_by_admin_location(nurses, "nurses")

if not patients.empty and "gender" in patients.columns:
    patients["gender_clean"] = patients["gender"].apply(normalize_gender)
    patients = patients[patients["gender_clean"].isin(["F
