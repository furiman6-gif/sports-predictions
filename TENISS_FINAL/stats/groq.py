import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def get_2026_matches():
    print("Pobieram najnowsze mecze 2026 z last_fifty.html...")
    r = requests.get("https://www.tennisabstract.com/charting/last_fifty.html", headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    
    matches = []
    for link in soup.find_all('a', href=True):
        text = link.get_text(strip=True)
        if re.search(r'2026-\d{2}-\d{2}', text) and ('(ATP)' in text or '(WTA)' in text):
            href = link['href']
            full_url = "https://www.tennisabstract.com/charting/" + href if not href.startswith('http') else href
            matches.append({
                'date': re.search(r'2026-\d{2}-\d{2}', text).group(0),
                'match_text': text,
                'url': full_url
            })
    print(f"Znaleziono {len(matches)} meczów z 2026")
    return matches

def extract_stats(url):
    try:
        r = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        stats = []
        # Szukamy tabeli STATS OVERVIEW
        for table in soup.find_all('table'):
            if "STATS OVERVIEW" not in table.get_text():
                continue
                
            for row in table.find_all('tr'):
                cells = row.find_all(['th', 'td'])
                if not cells or len(cells) < 8:
                    continue
                    
                first_cell = cells[0].get_text(strip=True)
                
                # Pomijamy nagłówki i sety
                if first_cell in ["STATS OVERVIEW", "SET 1", "SET 2", "SET 3"]:
                    continue
                    
                # To musi być wiersz zawodnika
                if len(first_cell.split()) >= 2:  
                    try:
                        values = [c.get_text(strip=True) for c in cells]
                        player = values[0]
                        
                        first_pct   = float(values[3].rstrip('%')) if '%' in values[3] else None
                        second_pct  = float(values[4].rstrip('%')) if '%' in values[4] else None
                        
                        # BPSaved np. "6/9"
                        bp_saved_pct = None
                        if '/' in values[5]:
                            try:
                                made, total = map(int, values[5].split('/'))
                                bp_saved_pct = round((made / total) * 100, 1) if total > 0 else None
                            except:
                                pass
                        
                        rpw_pct = float(values[6].rstrip('%')) if '%' in values[6] else None
                        
                        stats.append({
                            'player': player,
                            '1st_serve_pts_won_pct': first_pct,
                            '2nd_serve_pts_won_pct': second_pct,
                            'bp_saved_pct': bp_saved_pct,
                            'return_pts_won_pct': rpw_pct,
                            'url': url
                        })
                    except:
                        continue
        return stats
    except Exception as e:
        print(f"   BŁĄD przy {url}: {e}")
        return []

# ================== URUCHOMIENIE ==================
matches = get_2026_matches()
all_stats = []

for idx, m in enumerate(matches, 1):
    print(f"[{idx}/{len(matches)}] Przetwarzam → {m['date']} {m['match_text']}")
    match_stats = extract_stats(m['url'])
    for s in match_stats:
        s['date'] = m['date']
        s['match'] = m['match_text']
    all_stats.extend(match_stats)
    time.sleep(1)   # grzeczność wobec serwera

# Zapis do CSV
if all_stats:
    df = pd.DataFrame(all_stats)
    df_atp = df[df['match'].str.contains('ATP', na=False)].copy()
    df_wta = df[df['match'].str.contains('WTA', na=False)].copy()
    
    df_atp.to_csv('atp_2026_charted.csv', index=False, encoding='utf-8-sig')
    df_wta.to_csv('wta_2026_charted.csv', index=False, encoding='utf-8-sig')
    
    print("\n✅ ZROBIONE!")
    print(f"ATP → {len(df_atp)} wierszy (atp_2026_charted.csv)")
    print(f"WTA → {len(df_wta)} wierszy (wta_2026_charted.csv)")
else:
    print("Niestety nic nie wyciągnęło – daj znać, dam wersję z Selenium.")
