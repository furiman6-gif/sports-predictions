import pandas as pd
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import datetime as _dt
DATE = _dt.date.today().isoformat()
LEAGUE_FILTER = ""
PRED = "auto_outputs_future/future_predictions_1row_all_targets.csv"
FIX  = "fixtures_22_ligues_2025_26.csv"

df = pd.read_csv(PRED, low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
today = df[df['Date'] == DATE].copy()
if LEAGUE_FILTER:
    today = today[today['league'].str.contains(LEAGUE_FILTER, case=False, na=False)]

# dolacz godziny meczow z fixtures
if os.path.exists(FIX):
    fx = pd.read_csv(FIX, low_memory=False)
    if 'Date' in fx.columns and 'Time' in fx.columns:
        fx['Date'] = pd.to_datetime(fx['Date'], dayfirst=True, errors='coerce')
        # zmapuj nazwy druzyn (uproszczone - lookup po dacie + dowolnej druzynie)
        fx_today = fx[fx['Date'] == DATE][['HomeTeam','AwayTeam','Time']].copy()
    else:
        fx_today = pd.DataFrame()
else:
    fx_today = pd.DataFrame()

TARGETS = {
    "FTR":     ("FTR_pred_class",     "FTR_max_prob",     "FTR_test_acc"),
    "O25":     ("O25_pred_class",     "O25_max_prob",     "O25_test_acc"),
    "BTTS":    ("BTTS_pred_class",    "BTTS_max_prob",    "BTTS_test_acc"),
    "CRD25":   ("CRD25_pred_class",   "CRD25_max_prob",   "CRD25_test_acc"),
    "CORNOU":  ("CORNOU_pred_class",  "CORNOU_max_prob",  "CORNOU_test_acc"),
    "SHTOU":   ("SHTOU_pred_class",   "SHTOU_max_prob",   "SHTOU_test_acc"),
    "FOULSOU": ("FOULSOU_pred_class", "FOULSOU_max_prob", "FOULSOU_test_acc"),
    "HGOOU":   ("HGOOU_pred_class",   "HGOOU_max_prob",   "HGOOU_test_acc"),
    "AGOOU":   ("AGOOU_pred_class",   "AGOOU_max_prob",   "AGOOU_test_acc"),
}

def get_threshold(row, tname):
    col = f"{tname}_ou_threshold"
    return row[col] if col in row and pd.notna(row[col]) else ""

def fmt_pick(tname, cls, thr):
    if pd.isna(cls): return None
    cls = str(cls)
    if tname == "FTR":
        return {"H":"1 (gosp)", "D":"X (remis)", "A":"2 (gosc)"}.get(cls, cls)
    if cls == "O": return f"OVER {thr}"
    if cls == "U": return f"UNDER {thr}"
    if cls == "YES": return "TAK (BTTS)"
    if cls == "NO":  return "NIE (BTTS)"
    return cls

# zbierz wszystkie typy
all_tips = []
for _, r in today.iterrows():
    home, away, lg = r['HomeTeam'], r['AwayTeam'], r.get('league','')
    # godzina
    t = ""
    if not fx_today.empty:
        m = fx_today[(fx_today['HomeTeam'].str.contains(home[:6], case=False, na=False)) |
                     (fx_today['AwayTeam'].str.contains(away[:6], case=False, na=False))]
        if not m.empty: t = str(m.iloc[0]['Time'])
    for tname, (cls_c, prob_c, acc_c) in TARGETS.items():
        if cls_c not in r or pd.isna(r[cls_c]): continue
        prob = r.get(prob_c, 0); acc = r.get(acc_c, 0)
        if pd.isna(prob) or pd.isna(acc): continue
        thr = get_threshold(r, tname)
        pick = fmt_pick(tname, r[cls_c], thr)
        score = float(prob) * float(acc)
        all_tips.append({
            'time': t, 'liga': lg, 'mecz': f"{home} vs {away}",
            'target': tname, 'typ': pick,
            'prob': float(prob), 'acc': float(acc), 'score': score
        })

tips = pd.DataFrame(all_tips)
print(f"\n{'='*90}")
print(f"  NAJLEPSZE TYPY NA DZIS ({DATE}) - {len(today)} meczow, {len(tips)} typow")
print(f"{'='*90}\n")

# TOP 30 globalnie
top = tips.sort_values('score', ascending=False).head(30)
print("--- TOP 30 TYPOW (prob x accuracy) ---")
for _, t in top.iterrows():
    print(f"  [{t['time']:>5}] {t['liga']:<25} {t['mecz']:<45} | {t['target']:<8} {t['typ']:<14} prob={t['prob']:.2%} acc={t['acc']:.2%} score={t['score']:.3f}")

# TOP 5 per target
print(f"\n\n--- TOP 5 PER TARGET ---")
for tname in TARGETS:
    sub = tips[tips['target']==tname].sort_values('score', ascending=False).head(5)
    if sub.empty: continue
    print(f"\n[{tname}]")
    for _, t in sub.iterrows():
        print(f"  [{t['time']:>5}] {t['mecz']:<45} {t['typ']:<14} prob={t['prob']:.2%} acc={t['acc']:.2%}")

# zapisz csv
tips.sort_values('score', ascending=False).to_csv(f'typy_dzis_{DATE}.csv', index=False, encoding='utf-8-sig')
print(f"\n\nZapisano: typy_dzis_{DATE}.csv ({len(tips)} typow)")
