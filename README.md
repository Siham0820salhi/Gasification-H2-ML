# Prediction de la Production d'Hydrogene par Gazeification

## Description
Ce projet applique le Machine Learning pour predire la production de H2 (vol%)
lors de la gazeification de dechets a partir des conditions operatoires.

## Meilleur modele : Extra Trees — R²=0.8286 | MAE=3.095 vol%

## Structure du projet
```
GASIFICATION_EDA/
├── Gasification Dataset.csv    # Dataset brut
├── 01_nettoyage.ipynb          # Chargement et nettoyage
├── 02_eda.ipynb                # Analyse exploratoire
├── 03_feature_engineering.ipynb # Creation des 13 nouvelles features
├── 04_random_forest.ipynb      # Modele RF (R²=0.8205)
├── 05_extra_trees.ipynb        # Modele Extra Trees (R²=0.8286) ← MEILLEUR
├── 06_xgboost.ipynb            # Modele XGBoost (R²=0.7990)
├── 07_comparaison.ipynb        # Comparaison des 3 modeles
├── 08_shap.ipynb               # Interpretabilite SHAP
├── app.py                      # Interface Streamlit
├── models/
│   ├── model_rf.pkl
│   ├── model_extra_trees.pkl
│   └── model_xgb.pkl
└── requirements.txt
```

## Installation
```bash
pip install -r requirements.txt
```

## Ordre d'execution
1. `01_nettoyage.ipynb`
2. `02_eda.ipynb`
3. `03_feature_engineering.ipynb`
4. `04_random_forest.ipynb`    → genere models/model_rf.pkl
5. `05_extra_trees.ipynb`      → genere models/model_extra_trees.pkl
6. `06_xgboost.ipynb`          → genere models/model_xgb.pkl
7. `07_comparaison.ipynb`
8. `08_shap.ipynb`

## Lancer l'interface
```bash
streamlit run app.py
```
Ouvre automatiquement http://localhost:8501

## Resultats
| Modele        |  R²   |  MAE   | RMSE  | Std R² |
|---------------|-------|--------|-------|--------|
| Extra Trees   | 0.8286| 3.095  | 4.556 | 0.053  |
| Random Forest | 0.8205| 3.191  | 4.662 | 0.067  |
| XGBoost       | 0.7990| 3.391  | 4.933 | 0.058  |

## Features importantes (SHAP)
1. Cumul_Temp — energie thermique cumulee
2. Thermal_Work — travail thermique
3. Temp_squared — effet non-lineaire de la temperature
