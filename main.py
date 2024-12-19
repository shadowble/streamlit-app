import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import base64




# Configuration de la page avec un style personnalisé
st.set_page_config(
    page_title="Estimation Immobilière",
    page_icon="🏠",
    layout="wide"
)

# Styles CSS personnalisés
def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# CSS personnalisé pour un design bleu moderne
custom_css = """
<style>
    /* Couleurs de base */
    :root {
        --primary-blue: #3498db;
        --dark-blue: #2980b9;
        --light-blue: #87CEFA;
        --background-color: #f4f6f7;
    }

    /* Style du corps de la page */
    .stApp {
        background-color: var(--background-color);
    }

    /* Personnalisation des titres */
    .stTitle {
        color: var(--dark-blue);
        font-weight: bold;
    }

    /* Personnalisation des boutons */
    .stButton > button {
        background-color: var(--primary-blue);
        color: white !important;
        border-radius: 5px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: var(--dark-blue) !important;
        transform: scale(1.05);
    }

    /* Personnalisation de la sidebar */
    .css-1aumxhk {
        background-color: var(--light-blue);
    }

    /* Style des métriques */
    .stMetric > div {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
"""

# Injection du CSS personnalisé
st.markdown(custom_css, unsafe_allow_html=True)

# Fonction pour générer un logo en base64
def get_base64_logo():
    # Logo SVG simple et moderne
    svg_logo = f'''
    <svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
        <rect width="200" height="100" fill="#3498db" rx="10" ry="10"/>
        <text x="50%" y="50%" text-anchor="middle" font-family="Arial" font-size="20" fill="white">
            Data Immo
        </text>
    </svg>
    '''
    return base64.b64encode(svg_logo.encode('utf-8')).decode('utf-8')

# Charger les données
@st.cache_data
def load_data():
    data = pd.read_csv("valeur.csv", sep=";")
    label_encoder = LabelEncoder()
    data["Code departement num"] = label_encoder.fit_transform(data["Code departement"])
    return data, label_encoder

# Entraînement du modèle
def train_model(data):
    X = data[["Surface reelle bati","Code departement num"]].fillna(0)
    y = data["Valeur fonciere"].fillna(0)
    model = LinearRegression()
    model.fit(X, y)
    return model

# Fonction principale
def main():
    # Charger les données
    data, label_encoder = load_data()
    
    # Titre et logo
    col1, col2 = st.columns([1, 3])
    with col1:
        logo_base64 = get_base64_logo()
        st.markdown(f'<img src="data:image/svg+xml;base64,{logo_base64}" style="width:200px">', unsafe_allow_html=True)
    
    with col2:
        st.title("Estimation Immobilière Intelligente")
    
    # Sidebar avec design
    st.sidebar.header("🏘️ Paramètres de Recherche")
    
    # Filtres
    region = st.sidebar.selectbox(
        "Choisissez une région", 
        data["Code postal"].unique(),
        help="Sélectionnez le code postal de votre bien"
    )
    
    local = st.sidebar.selectbox(
        "Choisissez le type de local", 
        data["Type local"].unique(),
        help="Type de bien immobilier"
    )
    
    piece = st.sidebar.selectbox(
        "Nombre de pièces", 
        data["Nombre pieces principales"].unique(),
        help="Nombre de pièces principales"
    )
    
    # Filtrer les données
    donnees_filtrees = data[
        (data["Code postal"] == region) &
        (data["Type local"] == local) &
        (data["Nombre pieces principales"] == piece)
    ]
    
    # Tendances des prix
    st.subheader("🔍 Tendances des Prix Immobiliers")
    tendances_prix = donnees_filtrees.groupby("Surface reelle bati")["Valeur fonciere"].mean().reset_index()
    
    # Graphique interactif avec Plotly
    fig = px.bar(
        tendances_prix, 
        x="Surface reelle bati", 
        y="Valeur fonciere",
        labels={"Surface reelle bati": "Surface (m²)", "Valeur fonciere": "Prix Moyen (€)"},
        title="Prix Moyen par Surface"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Entraîner le modèle
    model = train_model(data)
    
    # Formulaire d'estimation
    st.header("💰 Estimation du Prix d'un Bien Immobilier")
    estimation = None
    
    with st.form("estimation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            surface = st.number_input(
                "Surface réelle bâtie (m²):", 
                min_value=1, 
                max_value=1000, 
                step=1,
                help="Entrez la surface totale du bien"
            )
        
        with col2:
            localisation = st.selectbox(
                "Département:", 
                options=data["Code departement num"].unique(),
                help="Sélectionnez le code du département"
            )
        
        type_bien = st.selectbox(
            "Type de bien:", 
            options=data["Type local"].unique(),
            help="Choisissez le type de bien immobilier"
        )
        
        submitted = st.form_submit_button("Estimer le Prix")
        
        if submitted:
            estimation = model.predict(np.array([[surface, localisation]]))[0]
            
            # Affichage de l'estimation avec design
            st.markdown(f"""
            <div style="
                background-color: #3498db; 
                color: white; 
                padding: 20px; 
                border-radius: 10px;
                text-align: center;
            ">
                <h3>Estimation du Bien</h3>
                <p>Prix estimé pour un bien de {surface} m² dans le département {localisation} ({type_bien}) :</p>
                <h2>{estimation:,.2f} €</h2>
            </div>
            """, unsafe_allow_html=True)

    if estimation is not None:
        simulation_result = pd.DataFrame({
        "Surface réelle bâtie": [surface],
        "Département": [localisation],
        "Type de bien": [type_bien],
        "Prix estimé (€)": [estimation]
    })
    
        simulation_result.to_excel("simulation_result.xlsx", index=False)

    st.download_button(
    label="Télécharger le résultat de la simulation",
    data=open("simulation_result.xlsx", "rb"),
    file_name="simulation_result.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)



# Exécution de l'application
if __name__ == "__main__":
    main()