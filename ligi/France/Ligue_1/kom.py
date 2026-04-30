import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, log_loss
import lightgbm as lgb

# 1. WCZYTANIE DANYCH
df = pd.read_csv(
    r'c:\Users\furim\Desktop\engfoot\ligi\France\Ligue_1\final_feature_engineered_v2.csv',
    low_memory=False
)

print("--- Dane wejściowe ---")
print(f"Rozmiar: {df.shape}")
print(f"Wyniki: {df['re_ult'].value_counts().to_dict()}")

# 2. PREPROCESSING

# Target: W=2, D=1, L=0
le_result = LabelEncoder()
df['Target'] = le_result.fit_transform(df['re_ult'])
print(f"Klasy: {dict(zip(le_result.classes_, le_result.transform(le_result.classes_)))}")

# Featury: wszystkie kolumny numeryczne (bez identyfikatorów i targetu)
drop_cols = ['Date', 'HomeTeam', 'AwayTeam', 're_ult', 'Target']
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].apply(lambda col: col.astype(str).str.replace(',', '.', regex=False)).astype(float)
y = df['Target']

# Podział chronologiczny (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# 3. TRENING LIGHTGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data  = lgb.Dataset(X_test,  label=y_test, reference=train_data)

params = {
    'objective':     'multiclass',
    'num_class':     3,
    'metric':        'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves':    63,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':  5,
    'is_unbalance':  True,
    'verbosity':    -1,
    'seed':          42
}

print("\n--- Trening LightGBM ---")
model = lgb.train(
    params,
    train_data,
    num_boost_round=300,
    valid_sets=[test_data],
    callbacks=[
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=30, verbose=True)
    ]
)

# 4. PREDYKCJA I METRYKI
y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
ll   = log_loss(y_test, y_prob, labels=[0, 1, 2])

# Brier Score
y_test_onehot = np.zeros((len(y_test), 3))
y_test_onehot[np.arange(len(y_test)), y_test] = 1
brier = np.mean(np.sum((y_prob - y_test_onehot) ** 2, axis=1))

print("\n--- WYNIKI EWALUACJI ---")
print(f"Accuracy:    {acc:.4f}")
print(f"Precision:   {prec:.4f}")
print(f"LogLoss:     {ll:.4f}")
print(f"Brier Score: {brier:.4f}")

# Top 10 najważniejszych cech
importance = pd.Series(model.feature_importance(importance_type='gain'), index=feature_cols)
print("\n--- Top 10 cech (gain) ---")
print(importance.nlargest(10).to_string())
