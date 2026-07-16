import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import math
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(
    page_title="Gazeification H2 Predictor",
    page_icon="⚗️",
    layout="wide"
)

# ── CSS original + ajouts historique ─────────────────────────────────────────
st.markdown("""
<style>
    .main-title {font-size:2rem; font-weight:600; color:#1B3A5C; margin-bottom:0}
    .subtitle   {font-size:1rem; color:#6C757D; margin-top:0; margin-bottom:1.5rem}
    .metric-box {background:#F5F6F7; border-radius:8px; padding:1rem; text-align:center}
    .metric-val {font-size:2rem; font-weight:600; color:#2E6DA4}
    .metric-lbl {font-size:0.8rem; color:#6C757D; margin-top:4px}
    .result-box {background:#E6F4EF; border:1px solid #1D7A5F; border-radius:10px;
                 padding:1.5rem; text-align:center; margin-top:1rem}
    .result-h2  {font-size:3rem; font-weight:700; color:#1D7A5F}
    .result-unit{font-size:1.2rem; color:#6C757D}
    .warn-box   {background:#FFF8E1; border-left:4px solid #BA7517;
                 padding:0.8rem 1rem; border-radius:4px; font-size:0.9rem}
    .info-box   {background:#E3F2FD; border-left:4px solid #1565C0;
                 padding:0.8rem 1rem; border-radius:4px; font-size:0.9rem}

    /* ── Ajouts historique ── */
    .hist-stat {background:#F5F6F7; border-radius:10px; padding:1rem;
                text-align:center; border:1px solid #E0E5EA}
    .hist-stat-val {font-size:1.6rem; font-weight:700; color:#1B3A5C}
    .hist-stat-lbl {font-size:0.75rem; color:#6C757D; margin-top:3px; text-transform:uppercase; letter-spacing:0.5px}
    .dash-card {background:#FFFFFF; border:1px solid #E0E5EA; border-radius:10px;
                padding:1rem 1.2rem; margin-bottom:0.8rem}
    .dash-title {font-size:0.7rem; font-weight:600; color:#2E6DA4;
                 text-transform:uppercase; letter-spacing:1px;
                 border-bottom:1px solid #EEF2F5; padding-bottom:6px; margin-bottom:10px}
    .best-exp {background:#E6F4EF; border:1px solid #1D7A5F; border-radius:8px;
               padding:0.8rem 1rem; font-size:0.88rem; color:#1B3A5C}
    .h2-pill {display:inline-block; padding:3px 12px; border-radius:20px;
              font-weight:600; font-size:0.8rem}
    .pill-ex  {background:#D4EDDA; color:#155724}
    .pill-gd  {background:#D1ECF1; color:#0C5460}
    .pill-md  {background:#FFF3CD; color:#856404}
    .pill-lw  {background:#F8D7DA; color:#721C24}
</style>
""", unsafe_allow_html=True)

# ── Session state pour historique ──────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Titre ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">⚗️ Gazeification H2 — Predicteur ML</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Approche Hybride : TWTS → Modele Global (R²=0.828) | Leather scraps → Modele Specialise (R²=0.859)</p>', unsafe_allow_html=True)
st.divider()

# ── Charger les modeles ───────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_global  = joblib.load('models/model_extra_trees.pkl') if os.path.exists('models/model_extra_trees.pkl') else None
    model_leather = joblib.load('models/model_leather.pkl')     if os.path.exists('models/model_leather.pkl')     else None
    return model_global, model_leather

model_global, model_leather = load_models()

if model_global is None:
    st.error("⚠️ Modele non trouve. Executez d'abord le notebook 05_extra_trees.ipynb.")
    st.stop()

@st.cache_data
def load_reference_data():
    """Charge et prétraite les données d'entraînement pour lookup exact des features."""
    try:
        df = pd.read_csv('data_clean.csv')
        g = df.groupby('Experiment_Group')
        max_t = g['Time'].transform('max')
        df['Relative_Time']         = df['Time'] / (max_t + 1e-6)
        df['Thermal_Work']          = df['Temperature'] * df['Time']
        df['Agent_Intensity']       = df['AgentFlow'] * df['ProcessTemperature']
        df['Thermal_Acceleration']  = df['Thermal_Work'] / (df['Time'] + 0.1)
        df['Relative_Severity']     = df['Relative_Time'] * df['ProcessTemperature']
        df['Kinetic_Agent_Density'] = df['AgentFlow'] * df['Relative_Time']
        df['Cumul_Temp']            = g['Temperature'].cumsum()
        df['Temp_Norm_in_Run']      = g['Temperature'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
        df['Temp_x_AgentFlow']      = df['Temperature'] * df['AgentFlow']
        df['Temp_x_CatalystRatio']  = df['Temperature'] * df['CatalystRatio']
        df['Cumul_x_AgentFlow']     = df['Cumul_Temp'] * df['AgentFlow']
        df['Temp_squared']          = df['Temperature'] ** 2
        df['Cumul_Temp_squared']    = df['Cumul_Temp'] ** 2
        df['Temp_mean_run']         = g['Temperature'].transform('mean')
        df['Temp_max_run']          = g['Temperature'].transform('max')
        df['Temp_std_run']          = g['Temperature'].transform('std').fillna(0)
        return df
    except Exception:
        return None

df_ref = load_reference_data()

# ── Sidebar — parametres ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parametres d'entree")

    st.subheader("Conditions thermiques")
    temp          = st.slider("Temperature (Feed_Temp_C)", 6, 1000, 500, step=1)
    proc_temp     = st.selectbox("Reactor_Temp_C (°C)", [700, 800, 900, 1000], index=1)
    time_min      = st.slider("Temps (min)", 0, 101, 50, step=1)

    st.divider()
    st.subheader("Agent gazéifiant")
    agent_type    = st.selectbox("Agent Type", ["Air", "Oxygen"], index=1)
    agent_flow    = st.slider("Agent Flow (L/min)", 0.015, 0.400, 0.100, step=0.005, format="%.3f")

    st.divider()
    st.subheader("Echantillon & Catalyseur")
    sample_type   = st.selectbox("Sample Type", ["TWTS", "Leather scraps"])
    catalyst_type = st.selectbox("Catalyst Type", ["No Catalyst", "Al-Ni", "Marble dust"], index=1)
    catalyst_ratio= st.slider("Catalyst Ratio (%)", 0, 20, 10, step=1)

    st.divider()
    st.subheader("📋 Info Experience")
    operator_name = st.text_input("Operateur", placeholder="Nom / Initiales")
    lab_location  = st.text_input("Laboratoire", placeholder="Salle / Code")
    exp_notes     = st.text_area("Notes", placeholder="Conditions speciales...", height=70)

# ── Feature Engineering ───────────────────────────────────────────────────────
_prediction_source = ['approx']  # 'exact' ou 'approx'

def build_features(temp, proc_temp, time_min, agent_type,
                   agent_flow, sample_type, catalyst_type, catalyst_ratio):
    """
    Construit le vecteur de features pour la prédiction.
    1) Cherche une correspondance exacte dans les données d'entraînement.
    2) Si non trouvé, utilise des approximations corrigées.
    """
    num_cols = [
        'Time','Temperature','ProcessTemperature','AgentFlow','CatalystRatio',
        'Relative_Time','Thermal_Work','Agent_Intensity','Thermal_Acceleration',
        'Relative_Severity','Kinetic_Agent_Density','Cumul_Temp','Temp_Norm_in_Run',
        'Temp_x_AgentFlow','Temp_x_CatalystRatio','Cumul_x_AgentFlow',
        'Temp_squared','Cumul_Temp_squared','Temp_mean_run','Temp_max_run','Temp_std_run',
    ]

    # ── 1. Lookup exact dans les données d'entraînement ───────────────────────
    if df_ref is not None:
        mask = (
            (df_ref['Time'] == time_min) &
            (df_ref['Temperature'] == temp) &
            (df_ref['ProcessTemperature'] == proc_temp) &
            (df_ref['AgentType'] == agent_type) &
            (np.abs(df_ref['AgentFlow'] - agent_flow) < 1e-6) &
            (df_ref['SampleType'] == sample_type) &
            (df_ref['CatalystType'] == catalyst_type) &
            (np.abs(df_ref['CatalystRatio'] - catalyst_ratio) < 0.1)
        )
        if mask.any():
            row = df_ref[mask].iloc[0]
            feat = {c: row[c] for c in num_cols}
            feat.update({'AgentType': agent_type, 'SampleType': sample_type,
                         'CatalystType': catalyst_type})
            _prediction_source[0] = 'exact'
            return pd.DataFrame([feat])

    # ── 2. Approximations corrigées (conditions nouvelles / articles) ──────────
    _prediction_source[0] = 'approx'

    # T° de départ typique des expériences (~15°C)
    start_temp      = 15.0

    relative_time   = time_min / (101.0 + 1e-6)
    thermal_work    = temp * time_min
    agent_intensity = agent_flow * proc_temp
    thermal_acc     = thermal_work / (time_min + 0.1)
    relative_sev    = relative_time * proc_temp
    kinetic_density = agent_flow * relative_time

    # Cumul_Temp : somme cumulée des températures (≠ temp × time)
    # Approximation : (T_départ + T_courante) / 2 × nb_pas (profil linéaire moyen)
    avg_temp        = (start_temp + temp) / 2.0
    cumul_temp      = avg_temp * time_min

    # Temp_Norm_in_Run : normalisation DANS l'expérience, pas globale
    # [T_départ … ProcessTemperature] — pas [6 … 1000]
    temp_norm       = float(np.clip((temp - start_temp) / (proc_temp - start_temp + 1e-6), 0.0, 1.0))

    # Temp_mean_run : moyenne de toutes les T° de l'expérience
    # Approximation : (T_départ + ProcessTemperature) / 2
    temp_mean_run   = (start_temp + float(proc_temp)) / 2.0

    # Temp_max_run : max de T° dans l'expérience ≈ ProcessTemperature
    temp_max_run    = float(proc_temp)

    # Temp_std_run : écart-type pour profil linéaire de start_temp à proc_temp
    temp_std_run    = (float(proc_temp) - start_temp) / (2.0 * math.sqrt(3.0))

    cumul_temp_sq   = cumul_temp ** 2

    return pd.DataFrame([{
        'Time': time_min, 'Temperature': temp, 'ProcessTemperature': proc_temp,
        'AgentFlow': agent_flow, 'CatalystRatio': catalyst_ratio,
        'Relative_Time': relative_time, 'Thermal_Work': thermal_work,
        'Agent_Intensity': agent_intensity, 'Thermal_Acceleration': thermal_acc,
        'Relative_Severity': relative_sev, 'Kinetic_Agent_Density': kinetic_density,
        'Cumul_Temp': cumul_temp, 'Temp_Norm_in_Run': temp_norm,
        'Temp_x_AgentFlow': temp * agent_flow, 'Temp_x_CatalystRatio': temp * catalyst_ratio,
        'Cumul_x_AgentFlow': cumul_temp * agent_flow, 'Temp_squared': temp ** 2,
        'Cumul_Temp_squared': cumul_temp_sq, 'Temp_mean_run': temp_mean_run,
        'Temp_max_run': temp_max_run, 'Temp_std_run': temp_std_run,
        'AgentType': agent_type, 'SampleType': sample_type, 'CatalystType': catalyst_type,
    }])

# ── Prediction ────────────────────────────────────────────────────────────────
X_pred = build_features(temp, proc_temp, time_min, agent_type,
                        agent_flow, sample_type, catalyst_type, catalyst_ratio)

# ── Choix du modèle selon le type de déchet (approche hybride) ───────────────
if sample_type == 'Leather scraps' and model_leather is not None:
    # Modèle spécialisé Leather scraps (R²=0.859) — sans colonne SampleType
    active_model      = model_leather
    active_model_name = 'Leather scraps (specialise, R²=0.859)'
    active_mae        = 2.630
    X_model = X_pred.drop(columns=['SampleType'])
else:
    # Modèle global pour TWTS (R²=0.828) — meilleur que le modèle TWTS seul
    active_model      = model_global
    active_model_name = 'TWTS / Global (R²=0.828)'
    active_mae        = 3.095
    X_model = X_pred

prediction = float(np.clip(active_model.predict(X_model)[0], 0, 40))
mae        = active_mae
low, high  = max(0, prediction - mae), prediction + mae

# Niveau
if   prediction >= 25: level_text="Excellente production"; level_color="#1D7A5F"; pill_cls="pill-ex"
elif prediction >= 15: level_text="Bonne production";       level_color="#0C6B8A"; pill_cls="pill-gd"
elif prediction >= 8:  level_text="Production moderee";     level_color="#856404"; pill_cls="pill-md"
else:                  level_text="Production faible";      level_color="#721C24"; pill_cls="pill-lw"

# ════════════════════════════════════════════════════════════════════════════
# TABS : Prediction | Historique | Dashboard
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔬 Prediction", "📋 Historique", "📊 Dashboard"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICTION (ton code original conservé)
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Prediction H2")
        st.markdown(f"""
        <div class="result-box">
            <div class="result-h2" style="color:{level_color}">{prediction:.2f}</div>
            <div class="result-unit">vol% H2</div>
            <div style="margin-top:8px;font-size:0.9rem;color:#6C757D">{level_text}</div>
            <div style="margin-top:4px;font-size:0.85rem;color:#6C757D">
                Plage probable : {low:.1f} — {high:.1f} vol% (±MAE)
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Indicateur source de la prédiction ────────────────────────────
        if _prediction_source[0] == 'exact':
            st.markdown("""
            <div style="background:#E6F4EF;border-left:4px solid #1D7A5F;
                        padding:0.6rem 1rem;border-radius:4px;font-size:0.85rem">
                ✅ <b>Données exactes</b> — conditions trouvées dans le dataset d'entraînement.
                Les features sont calculées exactement comme à l'entraînement.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#FFF8E1;border-left:4px solid #BA7517;
                        padding:0.6rem 1rem;border-radius:4px;font-size:0.85rem">
                ⚠️ <b>Approximation</b> — conditions nouvelles (hors dataset).
                Les features temporelles (Cumul_Temp, Temp_Norm, etc.) sont estimées.
                Écart possible vs données réelles.
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Bouton Enregistrer ─────────────────────────────────────────────
        if st.button("💾 Enregistrer cette experience", use_container_width=True, type="primary"):
            entry = {
                "id":        len(st.session_state.history) + 1,
                "date":      datetime.now().strftime("%Y-%m-%d"),
                "heure":     datetime.now().strftime("%H:%M"),
                "operateur": operator_name or "—",
                "labo":      lab_location  or "—",
                "temp":      temp,
                "proc_temp": proc_temp,
                "time":      time_min,
                "agent":     agent_type,
                "flow":      round(agent_flow, 3),
                "sample":    sample_type,
                "catalyst":  catalyst_type,
                "cat_ratio": catalyst_ratio,
                "h2":        round(prediction, 3),
                "niveau":    level_text,
                "pill":      pill_cls,
                "low":       round(low, 2),
                "high":      round(high, 2),
                "notes":     exp_notes or "—",
            }
            st.session_state.history.append(entry)
            st.success(f"✅ Experience #{entry['id']} enregistree — H2 = {prediction:.2f} vol%")

        st.divider()
        st.subheader(f"Performance — {active_model_name}")
        if sample_type == 'Leather scraps' and model_leather is not None:
            perf_vals = [("0.8594","R²"),("2.630","MAE (vol%)"),("3.945","RMSE"),("0.027","Std R²")]
        else:
            perf_vals = [("0.8280","R²"),("3.095","MAE (vol%)"),("4.556","RMSE"),("0.055","Std R²")]
        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, (val, lbl) in zip([mc1,mc2,mc3,mc4], perf_vals):
            col.markdown(
                f'<div class="metric-box"><div class="metric-val">{val}</div>'
                f'<div class="metric-lbl">{lbl}</div></div>',
                unsafe_allow_html=True)

    with col2:
        st.subheader("Features calculees automatiquement")
        _x = X_pred.iloc[0]
        fe_data = {
            'Feature':    ['Cumul_Temp','Thermal_Work','Temp_squared',
                           'Relative_Time','Thermal_Acceleration','Temp_Norm_in_Run'],
            'Valeur':     [f"{_x['Cumul_Temp']:.0f}", f"{_x['Thermal_Work']:.0f}",
                           f"{_x['Temp_squared']:.0f}", f"{_x['Relative_Time']:.3f}",
                           f"{_x['Thermal_Acceleration']:.2f}", f"{_x['Temp_Norm_in_Run']:.3f}"],
            'Importance': ['#1 ⭐','#2 ⭐','#3 ⭐','#4','#5','#6']
        }
        st.dataframe(pd.DataFrame(fe_data), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Recap conditions")
        for k, v in {
            "Temperature": f"{temp} °C", "Reactor_Temp": f"{proc_temp} °C",
            "Temps": f"{time_min} min", "Agent": agent_type,
            "Agent Flow": f"{agent_flow:.3f} L/min", "Dechet": sample_type,
            "Catalyseur": catalyst_type, "Ratio cat.": f"{catalyst_ratio} %",
            "Operateur": operator_name or "—", "Laboratoire": lab_location or "—",
        }.items():
            st.markdown(f"**{k}** : {v}")

    st.divider()
    st.subheader("Comparaison des 3 modeles")
    comp_df = pd.DataFrame({
        'Modele':        ['Global (TWTS)', 'Leather specialise', 'TWTS seul (ref)'],
        'Materiau':      ['TWTS', 'Leather scraps', 'TWTS'],
        'R²':            [0.8280, 0.8594, 0.7853],
        'MAE (vol%)':    [3.095,  2.630,  3.530],
        'RMSE':          [4.556,  3.945,  5.216],
        'Std R²':        [0.055,  0.027,  0.064],
        'Statut':        ['Actif TWTS', 'Actif Leather', 'Non utilise'],
    })
    st.dataframe(
        comp_df.style
               .highlight_max(subset=['R²'], color='#E6F4EF')
               .highlight_min(subset=['MAE (vol%)','RMSE','Std R²'], color='#E6F4EF')
               .format({'R²':'{:.4f}','MAE (vol%)':'{:.3f}','RMSE':'{:.3f}','Std R²':'{:.3f}'}),
        use_container_width=True, hide_index=True)
    st.markdown("""
    <div class="info-box">
        💡 <b>Approche Hybride</b> : TWTS utilise le modele global (R²=0.828) car il beneficie
        des patterns communs aux deux dechets. Leather scraps utilise son modele specialise
        (R²=0.859, MAE=2.63) — gain de +3% R² et -0.46 vol% MAE par rapport au global.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — HISTORIQUE
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    hist = st.session_state.history

    if not hist:
        st.info("Aucune experience enregistree. Allez dans l'onglet **🔬 Prediction** et cliquez **Enregistrer cette experience**.")
    else:
        # Stats rapides
        h2_vals = [e['h2'] for e in hist]
        sc1, sc2, sc3, sc4 = st.columns(4)
        for col, val, lbl, color in [
            (sc1, len(hist),           "Experiences",   "#1B3A5C"),
            (sc2, f"{max(h2_vals):.2f}","Meilleur H2",  "#1D7A5F"),
            (sc3, f"{np.mean(h2_vals):.2f}","Moyenne H2","#2E6DA4"),
            (sc4, f"{min(h2_vals):.2f}","Minimum H2",   "#BA7517"),
        ]:
            col.markdown(
                f'<div class="hist-stat">'
                f'<div class="hist-stat-val" style="color:{color}">{val}</div>'
                f'<div class="hist-stat-lbl">{lbl}</div></div>',
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tableau historique
        rows_html = ""
        for e in reversed(hist):
            rows_html += f"""
            <tr style="border-bottom:1px solid #F0F4F8">
              <td style="padding:8px 10px;font-weight:600;color:#2E6DA4">#{e['id']}</td>
              <td style="padding:8px 10px">{e['date']}</td>
              <td style="padding:8px 10px">{e['heure']}</td>
              <td style="padding:8px 10px">{e['operateur']}</td>
              <td style="padding:8px 10px">{e['labo']}</td>
              <td style="padding:8px 10px">{e['temp']}°C</td>
              <td style="padding:8px 10px">{e['proc_temp']}°C</td>
              <td style="padding:8px 10px">{e['time']} min</td>
              <td style="padding:8px 10px">{e['agent']}</td>
              <td style="padding:8px 10px">{e['flow']}</td>
              <td style="padding:8px 10px">{e['sample']}</td>
              <td style="padding:8px 10px">{e['catalyst']}</td>
              <td style="padding:8px 10px">{e['cat_ratio']}%</td>
              <td style="padding:8px 10px">
                <span class="h2-pill {e['pill']}">{e['h2']:.2f} vol%</span>
              </td>
              <td style="padding:8px 10px;font-size:0.75rem;color:#6C757D">
                {str(e['notes'])[:25]}{'…' if len(str(e['notes']))>25 else ''}
              </td>
            </tr>"""

        st.markdown(f"""
        <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;font-size:0.82rem;background:#fff;border-radius:10px;border:1px solid #E0E5EA">
          <thead>
            <tr style="background:#1B3A5C;color:#fff">
              <th style="padding:9px 10px;text-align:left">#</th>
              <th style="padding:9px 10px;text-align:left">Date</th>
              <th style="padding:9px 10px;text-align:left">Heure</th>
              <th style="padding:9px 10px;text-align:left">Operateur</th>
              <th style="padding:9px 10px;text-align:left">Labo</th>
              <th style="padding:9px 10px;text-align:left">T feed</th>
              <th style="padding:9px 10px;text-align:left">T react</th>
              <th style="padding:9px 10px;text-align:left">Temps</th>
              <th style="padding:9px 10px;text-align:left">Agent</th>
              <th style="padding:9px 10px;text-align:left">Flow</th>
              <th style="padding:9px 10px;text-align:left">Dechet</th>
              <th style="padding:9px 10px;text-align:left">Catalyseur</th>
              <th style="padding:9px 10px;text-align:left">Cat%</th>
              <th style="padding:9px 10px;text-align:left">H2 predit</th>
              <th style="padding:9px 10px;text-align:left">Notes</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Actions
        act1, act2, act3 = st.columns(3)
        with act1:
            csv = pd.DataFrame(hist).to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Exporter CSV", csv,
                f"experiences_H2_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv", use_container_width=True)
        with act2:
            if st.button("🗑️ Effacer tout l'historique", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        with act3:
            best = max(hist, key=lambda x: x['h2'])
            st.markdown(
                f'<div class="best-exp">🏆 Meilleure exp : <b>#{best["id"]}</b><br>'
                f'H2 = <b>{best["h2"]:.2f} vol%</b><br>'
                f'{best["temp"]}°C · {best["agent"]} · {best["sample"]}</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    hist = st.session_state.history

    if len(hist) < 2:
        st.info("Enregistrez au moins **2 experiences** pour voir les graphiques du dashboard.")
    else:
        df_h   = pd.DataFrame(hist)
        h2_vals = df_h['h2'].tolist()

        # KPIs
        kc1, kc2, kc3, kc4 = st.columns(4)
        for col, val, lbl, color in [
            (kc1, f"{max(h2_vals):.2f}", "Meilleur H2 (vol%)", "#1D7A5F"),
            (kc2, f"{np.mean(h2_vals):.2f}", "Moyenne H2 (vol%)", "#2E6DA4"),
            (kc3, f"{np.std(h2_vals):.2f}", "Ecart-type", "#BA7517"),
            (kc4, str(len(hist)), "Total experiences", "#1B3A5C"),
        ]:
            col.markdown(
                f'<div class="hist-stat"><div class="hist-stat-val" style="color:{color}">{val}</div>'
                f'<div class="hist-stat-lbl">{lbl}</div></div>',
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        clrs = ["#1D7A5F" if v>=25 else "#2E6DA4" if v>=15 else "#BA7517" if v>=8 else "#C62828"
                for v in h2_vals]

        gc1, gc2 = st.columns(2)

        # Graphe 1 — H2 par experience
        with gc1:
            st.markdown('<div class="dash-card"><div class="dash-title">H2 predit par experience</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            fig.patch.set_facecolor('white'); ax.set_facecolor('white')
            ids  = [f"#{e['id']}" for e in hist]
            bars = ax.bar(ids, h2_vals, color=clrs, edgecolor='white', alpha=0.88, width=0.6)
            ax.axhline(np.mean(h2_vals), color='#2E6DA4', lw=1.8, ls='--',
                       label=f'Moy={np.mean(h2_vals):.1f}')
            for bar, val in zip(bars, h2_vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8, color='#1B3A5C')
            ax.set_ylabel("H2 (vol%)", fontsize=9, color='#6C757D')
            ax.set_xlabel("Experience", fontsize=9, color='#6C757D')
            ax.tick_params(colors='#6C757D', labelsize=8)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='#C0D0E0')
            ax.legend(fontsize=9)
            plt.tight_layout(pad=0.4)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        # Graphe 2 — H2 vs Temperature
        with gc2:
            st.markdown('<div class="dash-card"><div class="dash-title">H2 vs Temperature de chauffe</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            fig.patch.set_facecolor('white'); ax.set_facecolor('white')
            sc = ax.scatter(df_h['temp'].tolist(), h2_vals,
                            c=h2_vals, cmap='RdYlGn', s=80,
                            edgecolors='#1B3A5C', linewidths=0.5, alpha=0.85, zorder=3)
            plt.colorbar(sc, ax=ax, label='H2 (vol%)', shrink=0.8)
            for i, (t, h) in enumerate(zip(df_h['temp'].tolist(), h2_vals)):
                ax.annotate(f'#{hist[i]["id"]}', (t, h), textcoords="offset points",
                            xytext=(5, 3), fontsize=7.5, color='#1B3A5C')
            ax.set_xlabel("Temperature (°C)", fontsize=9, color='#6C757D')
            ax.set_ylabel("H2 (vol%)", fontsize=9, color='#6C757D')
            ax.tick_params(colors='#6C757D', labelsize=8)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.grid(True, linestyle='--', alpha=0.3, color='#C0D0E0')
            plt.tight_layout(pad=0.4)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        gc3, gc4 = st.columns(2)

        # Graphe 3 — H2 par agent
        with gc3:
            st.markdown('<div class="dash-card"><div class="dash-title">H2 par type d\'agent</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            fig.patch.set_facecolor('white'); ax.set_facecolor('white')
            agents   = df_h['agent'].unique()
            data_ag  = [df_h[df_h['agent']==a]['h2'].tolist() for a in agents]
            bp = ax.boxplot(data_ag, tick_labels=agents, patch_artist=True,
                            boxprops=dict(facecolor='#D1ECF1', color='#2E6DA4'),
                            medianprops=dict(color='#1D7A5F', linewidth=2.5),
                            whiskerprops=dict(color='#6C757D'),
                            capprops=dict(color='#6C757D'),
                            flierprops=dict(marker='o', markersize=4, alpha=0.5))
            if len(agents) > 1:
                bp['boxes'][1].set_facecolor('#D4EDDA')
            ax.set_ylabel("H2 (vol%)", fontsize=9, color='#6C757D')
            ax.tick_params(colors='#6C757D', labelsize=9)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='#C0D0E0')
            plt.tight_layout(pad=0.4)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        # Graphe 4 — Evolution H2
        with gc4:
            st.markdown('<div class="dash-card"><div class="dash-title">Evolution H2 dans le temps</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            fig.patch.set_facecolor('white'); ax.set_facecolor('white')
            xids = [e['id'] for e in hist]
            ax.plot(xids, h2_vals, color='#2E6DA4', lw=2, marker='o', markersize=6,
                    markerfacecolor='white', markeredgecolor='#2E6DA4', markeredgewidth=2)
            ax.fill_between(xids, [v-3.095 for v in h2_vals], [v+3.095 for v in h2_vals],
                            alpha=0.12, color='#2E6DA4', label='±MAE')
            ax.axhline(np.mean(h2_vals), color='#BA7517', lw=1.5, ls='--',
                       label=f'Moy={np.mean(h2_vals):.1f}', alpha=0.8)
            ax.set_xlabel("Experience #", fontsize=9, color='#6C757D')
            ax.set_ylabel("H2 predit (vol%)", fontsize=9, color='#6C757D')
            ax.tick_params(colors='#6C757D', labelsize=8)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.grid(True, linestyle='--', alpha=0.3, color='#C0D0E0')
            ax.legend(fontsize=9)
            plt.tight_layout(pad=0.4)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        # Meilleure experience
        best = max(hist, key=lambda x: x['h2'])
        st.markdown(f"""
        <div class="best-exp" style="margin-top:0.5rem">
          🏆 <b>Meilleure experience — #{best['id']}</b> &nbsp;|&nbsp;
          H2 = <b>{best['h2']:.2f} vol%</b> &nbsp;|&nbsp;
          {best['temp']}°C · {best['agent']} · {best['flow']} L/min ·
          {best['sample']} · {best['catalyst']} {best['cat_ratio']}% ·
          {best['time']} min &nbsp;|&nbsp;
          {best['operateur']} @ {best['labo']}
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Projet ML — Prediction H2 par Gazeification | Extra Trees | R²=0.8286 | GroupKFold(n=5) | 59 experiences")
