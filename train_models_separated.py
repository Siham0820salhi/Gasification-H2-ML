"""
Entraînement de deux modèles séparés :
  - model_twts.pkl         → uniquement sur TWTS
  - model_leather.pkl      → uniquement sur Leather scraps
"""
import numpy as np
import pandas as pd
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Chargement & Feature Engineering ─────────────────────────────────────────
df_all = pd.read_csv('Gasification Dataset.csv', delimiter=';')
df_all.columns = df_all.columns.str.strip()
for c in df_all.select_dtypes('object').columns:
    df_all[c] = df_all[c].str.strip()
df_all.loc[df_all['Hydrogen'] < 0, 'Hydrogen'] = 0.0

group_cols = ['ProcessTemperature','AgentType','AgentFlow',
              'SampleType','CatalystType','CatalystRatio']
df_all['Experiment_Group'] = df_all.groupby(group_cols).ngroup()
df_all = df_all.sort_values(['Experiment_Group','Time']).reset_index(drop=True)

rng = np.random.default_rng(42)
shuffled_ids = rng.permutation(df_all['Experiment_Group'].unique())
remap = {old: new for new, old in enumerate(shuffled_ids)}
df_all['Experiment_Group'] = df_all['Experiment_Group'].map(remap)

def add_features(df):
    g     = df.groupby('Experiment_Group')
    max_t = g['Time'].transform('max')
    df = df.copy()
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

df_all = add_features(df_all)

num_cols = [
    'Time','Temperature','ProcessTemperature','AgentFlow','CatalystRatio',
    'Relative_Time','Thermal_Work','Agent_Intensity','Thermal_Acceleration',
    'Relative_Severity','Kinetic_Agent_Density','Cumul_Temp','Temp_Norm_in_Run',
    'Temp_x_AgentFlow','Temp_x_CatalystRatio','Cumul_x_AgentFlow',
    'Temp_squared','Cumul_Temp_squared','Temp_mean_run','Temp_max_run','Temp_std_run',
]
cat_cols = ['AgentType','CatalystType']   # SampleType retiré (un seul matériau par modèle)

param_dist = {
    'regressor__n_estimators':      [300, 500, 700, 1000],
    'regressor__max_depth':         [6, 8, 10, 15, None],
    'regressor__min_samples_split': [2, 3, 5],
    'regressor__min_samples_leaf':  [2, 3, 5],
    'regressor__max_features':      ['sqrt', 'log2', 0.4, 0.6, 0.8],
}

os.makedirs('models', exist_ok=True)
results = {}

# ══════════════════════════════════════════════════════════════════════════════
# Fonction d'entraînement générique
# ══════════════════════════════════════════════════════════════════════════════
def train_model(sample_name, n_folds, model_path):
    print(f"\n{'='*65}")
    print(f"  MODELE — {sample_name.upper()}")
    print(f"{'='*65}")

    df = df_all[df_all['SampleType'] == sample_name].copy()
    n_groups = df['Experiment_Group'].nunique()
    print(f"Lignes : {len(df)}  |  Experiences : {n_groups}  |  Folds : {n_folds}")

    X      = df[num_cols + cat_cols]
    y      = df['Hydrogen']
    groups = df['Experiment_Group']

    preprocessor = ColumnTransformer(transformers=[
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

    base_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', ExtraTreesRegressor(random_state=42, n_jobs=-1))
    ])

    # Tuning
    print(f"Tuning (30 combinaisons, {n_folds}-fold)...")
    search = RandomizedSearchCV(
        base_pipe, param_distributions=param_dist,
        n_iter=30, scoring='r2',
        cv=GroupKFold(n_splits=n_folds),
        random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X, y, groups=groups)
    best_params = {k.replace('regressor__',''): v
                   for k, v in search.best_params_.items()
                   if k.startswith('regressor__')}
    print(f"R2 CV tuning = {search.best_score_:.4f}")
    for k, v in best_params.items():
        print(f"   {k:25s} = {v}")

    # Evaluation finale
    best_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', ExtraTreesRegressor(**best_params, random_state=42, n_jobs=-1))
    ])

    gkf = GroupKFold(n_splits=n_folds)
    all_y_true, all_y_pred = [], []
    fold_r2 = []

    print(f"\nEvaluation {n_folds} folds...")
    print("-"*65)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        best_pipe.fit(X_tr, y_tr)
        preds = np.clip(best_pipe.predict(X_te), 0, None)
        all_y_true.extend(y_te.values)
        all_y_pred.extend(preds)
        fr2  = r2_score(y_te, preds)
        fmae = mean_absolute_error(y_te, preds)
        fold_r2.append(fr2)
        print(f"Fold {fold+1} | R²={fr2:.4f}  MAE={fmae:.3f}")

    global_r2   = r2_score(all_y_true, all_y_pred)
    global_mae  = mean_absolute_error(all_y_true, all_y_pred)
    global_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    std_r2      = np.std(fold_r2)

    # Train score (diagnostic overfitting)
    best_pipe.fit(X, y)
    pred_train  = best_pipe.predict(X)
    r2_train    = r2_score(y, pred_train)
    mae_train   = mean_absolute_error(y, pred_train)

    print(f"\n{'='*65}")
    print(f"  R2   Test  (CV)   : {global_r2:.4f}")
    print(f"  R2   Train        : {r2_train:.4f}   (ecart overfitting = {r2_train-global_r2:.4f})")
    print(f"  MAE  Test  (CV)   : {global_mae:.4f} vol%")
    print(f"  MAE  Train        : {mae_train:.4f} vol%")
    print(f"  RMSE Test  (CV)   : {global_rmse:.4f} vol%")
    print(f"  Std R2 folds      : {std_r2:.4f}")
    print(f"{'='*65}")

    joblib.dump(best_pipe, model_path)
    print(f"Modele sauvegarde : {model_path}")

    results[sample_name] = {
        'r2_test': global_r2, 'r2_train': r2_train,
        'mae_test': global_mae, 'mae_train': mae_train,
        'rmse': global_rmse, 'std_r2': std_r2,
        'n_groups': n_groups
    }

# ── Entraînement ──────────────────────────────────────────────────────────────
train_model('TWTS',          n_folds=5, model_path='models/model_twts.pkl')
train_model('Leather scraps', n_folds=3, model_path='models/model_leather.pkl')

# ── Comparaison finale ────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  COMPARAISON FINALE")
print(f"{'='*65}")
print(f"{'Modele':<20} {'Exp':>4} {'R2_Test':>8} {'R2_Train':>9} {'Ecart':>7} {'MAE_Test':>9}")
print("-"*65)

# Modele global reference
print(f"{'Global (ref)':20} {'59':>4} {'0.8280':>8} {'0.9745':>9} {'0.1465':>7} {'3.095':>9}")

for name, r in results.items():
    gap = r['r2_train'] - r['r2_test']
    print(f"{name:<20} {r['n_groups']:>4} {r['r2_test']:>8.4f} "
          f"{r['r2_train']:>9.4f} {gap:>7.4f} {r['mae_test']:>9.4f}")

print(f"\nModeles sauvegardes dans models/")
print("  - model_twts.pkl")
print("  - model_leather.pkl")
