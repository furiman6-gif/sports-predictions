#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sofascore Football Data Scraper - FINAL v3
Strategy: uzywa strzalki "prev round" do przejscia przez wszystkie rundy
"""

import sys
import io

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except:
    pass

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==================== KONFIGURACJA ====================

SEASONS_TO_SCRAPE = ['25/26', '24/25', '23/24', '22/23', '21/22', '20/21', '19/20', '18/19']
MAX_ROUNDS = 60     # zabezpieczenie - niektore ligi maja duzo rund
PAGE_WAIT = 5       # czas na zaladowanie strony
ACTION_WAIT = 1.8   # czas po kliknieciu

DATA_DIR = Path("./data")
CHECKPOINT_FILE = DATA_DIR / "checkpoint.json"
MATCHES_DIR = DATA_DIR / "matches"

TOURNAMENTS = {
    "E0":  {"name": "Premier League",    "url": "https://www.sofascore.com/tournament/football/england/premier-league/17"},
    "E1":  {"name": "Championship",      "url": "https://www.sofascore.com/tournament/football/england/championship/18"},
    "E2":  {"name": "League One",        "url": "https://www.sofascore.com/tournament/football/england/league-one/19"},
    "E3":  {"name": "League Two",        "url": "https://www.sofascore.com/tournament/football/england/league-two/20"},
    "EC":  {"name": "National League",   "url": "https://www.sofascore.com/tournament/football/england/national-league/21"},
    "D1":  {"name": "Bundesliga",        "url": "https://www.sofascore.com/tournament/football/germany/bundesliga/35"},
    "D2":  {"name": "2. Bundesliga",     "url": "https://www.sofascore.com/tournament/football/germany/2-bundesliga/36"},
    "SP1": {"name": "La Liga",           "url": "https://www.sofascore.com/tournament/football/spain/laliga/8"},
    "SP2": {"name": "Segunda",           "url": "https://www.sofascore.com/tournament/football/spain/segunda-division/9"},
    "I1":  {"name": "Serie A",           "url": "https://www.sofascore.com/tournament/football/italy/serie-a/23"},
    "I2":  {"name": "Serie B",           "url": "https://www.sofascore.com/tournament/football/italy/serie-b/24"},
    "F1":  {"name": "Ligue 1",           "url": "https://www.sofascore.com/tournament/football/france/ligue-1/34"},
    "F2":  {"name": "Ligue 2",           "url": "https://www.sofascore.com/tournament/football/france/ligue-2/61"},
    "SC0": {"name": "Premiership SCO",   "url": "https://www.sofascore.com/tournament/football/scotland/premiership/40"},
    "SC1": {"name": "Championship SCO",  "url": "https://www.sofascore.com/tournament/football/scotland/championship/41"},
    "SC2": {"name": "League One SCO",    "url": "https://www.sofascore.com/tournament/football/scotland/league-one/42"},
    "SC3": {"name": "League Two SCO",    "url": "https://www.sofascore.com/tournament/football/scotland/league-two/43"},
    "B1":  {"name": "First Division A",  "url": "https://www.sofascore.com/tournament/football/belgium/first-division-a/37"},
    "N1":  {"name": "Eredivisie",        "url": "https://www.sofascore.com/tournament/football/netherlands/eredivisie/32"},
    "P1":  {"name": "Primeira Liga",     "url": "https://www.sofascore.com/tournament/football/portugal/primeira-liga/62"},
    "G1":  {"name": "Super League",      "url": "https://www.sofascore.com/tournament/football/greece/super-league/203"},
    "T1":  {"name": "Super Lig",         "url": "https://www.sofascore.com/tournament/football/turkey/super-lig/52"},
    "CUP_ENG": {"name": "FA Cup",        "url": "https://www.sofascore.com/tournament/football/england/fa-cup/25"},
    "CUP_GER": {"name": "DFB-Pokal",     "url": "https://www.sofascore.com/tournament/football/germany/dfb-pokal/26"},
    "CUP_ESP": {"name": "Copa del Rey",  "url": "https://www.sofascore.com/tournament/football/spain/copa-del-rey/11"},
    "CUP_ITA": {"name": "Coppa Italia",  "url": "https://www.sofascore.com/tournament/football/italy/coppa-italia/28"},
    "CUP_FRA": {"name": "Coupe de France","url": "https://www.sofascore.com/tournament/football/france/coupe-de-france/51"},
    "UCL":     {"name": "Champions League","url": "https://www.sofascore.com/tournament/football/europe/uefa-champions-league/7"},
    "UEL":     {"name": "Europa League", "url": "https://www.sofascore.com/tournament/football/europe/uefa-europa-league/679"},
}

# ==================== CHECKPOINT ====================

class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()

    def _load(self) -> Dict:
        if self.path.exists():
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data.setdefault("done_keys", [])
                data.setdefault("matches_count", 0)
                logger.info(f"Checkpoint loaded: {data['matches_count']} matches, {len(data['done_keys'])} seasons done")
                return data
            except Exception as e:
                logger.warning(f"CP load failed: {e}")
        return {"done_keys": [], "matches_count": 0,
                "start_time": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat()}

    def save(self):
        self.data["last_update"] = datetime.now().isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def is_done(self, key): return key in self.data["done_keys"]

    def mark_done(self, key, count):
        if key not in self.data["done_keys"]:
            self.data["done_keys"].append(key)
        self.data["matches_count"] += count
        self.save()

# ==================== SCRAPER ====================

# JavaScript helper: wszystkie wazne operacje sa w jednym skrypcie
JS_SELECT_SEASON = """
return (async (target) => {
    const ds = Array.from(document.querySelectorAll('button.dropdown__button'));
    const seasonBtn = ds.find(d => /^\\d{2}\\/\\d{2}$/.test(d.textContent.trim()));
    if (!seasonBtn) return 'no_btn';
    if (seasonBtn.textContent.trim() === target) return 'already';
    seasonBtn.click();
    await new Promise(r => setTimeout(r, 1000));
    const opts = Array.from(document.querySelectorAll('li[role="option"]'));
    const opt = opts.find(o => o.textContent.trim() === target);
    if (!opt) { document.body.click(); return 'no_option'; }
    opt.click();
    await new Promise(r => setTimeout(r, 2500));
    return 'ok';
})(arguments[0]);
"""

JS_GET_CURRENT_ROUND = """
// Zwraca numer obecnej rundy - szuka pierwszego dropdownu 'Round X' (ten w lewym panelu)
const leftPanel = document.querySelector('[class*="MatchesList"]') || document;
const dropdowns = Array.from(leftPanel.querySelectorAll('button.dropdown__button'));
const roundBtn = dropdowns.find(d => /^Round \\d+$/.test(d.textContent.trim()));
if (!roundBtn) return null;
const m = roundBtn.textContent.trim().match(/Round (\\d+)/);
return m ? parseInt(m[1]) : null;
"""

JS_CLICK_PREV_ROUND = """
// Znajdz przycisk strzalki prev (M6 11.99) obok dropdown Round w LEWYM panelu
const allDropdowns = Array.from(document.querySelectorAll('button.dropdown__button'));
let targetDropdown = null;
for (const d of allDropdowns) {
    if (/^Round \\d+$/.test(d.textContent.trim())) {
        targetDropdown = d;
        break; // weź pierwszy
    }
}
if (!targetDropdown) return 'no_dropdown';

const parent = targetDropdown.parentElement.parentElement;
const buttons = Array.from(parent.querySelectorAll('button'));
const prevBtn = buttons.find(b => !b.classList.contains('dropdown__button') && b.innerHTML.includes('M6 11.99'));
if (!prevBtn) return 'no_prev';
if (prevBtn.disabled) return 'disabled';
prevBtn.click();
return 'ok';
"""

JS_COLLECT_MATCHES = """
const links = document.querySelectorAll('a[href*="/football/match/"]');
const seen = new Set();
const out = [];
for (const a of links) {
    const url = a.href;
    if (seen.has(url)) continue;
    seen.add(url);
    let mid = null;
    if (url.includes('#id:')) mid = url.split('#id:')[1];
    out.push({
        match_id: mid,
        url: url,
        text: (a.textContent || '').trim().substring(0, 300)
    });
}
return out;
"""


class SofascoreScraper:
    def __init__(self):
        opts = Options()
        opts.add_argument("--start-maximized")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)
        self.driver = webdriver.Chrome(options=opts)
        self.driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"}
        )
        logger.info("WebDriver ready")

    def get_season_matches(self, url: str, season: str) -> List[Dict]:
        try:
            self.driver.get(url)
            time.sleep(PAGE_WAIT)

            # Wybierz sezon
            res = self.driver.execute_script(JS_SELECT_SEASON, season)
            if res == 'no_option':
                logger.info(f"    Season {season} not available")
                return []
            if res == 'no_btn':
                logger.warning(f"    No season button")
                return []
            logger.info(f"    Season select: {res}")
            time.sleep(ACTION_WAIT + 1)

            return self._iterate_all_rounds()

        except Exception as e:
            logger.error(f"    Error: {e}")
            return []

    def _iterate_all_rounds(self) -> List[Dict]:
        """Od obecnej rundy klika strzalke '<' az do Round 1"""
        all_matches = {}  # url -> match_data

        # Najpierw zbierz z biezacej rundy
        current = self.driver.execute_script(JS_GET_CURRENT_ROUND)
        if current is None:
            # To puchar lub inny format
            logger.info(f"    Cup/knockout format (no rounds)")
            matches = self.driver.execute_script(JS_COLLECT_MATCHES) or []
            for m in matches:
                all_matches[m['url']] = m
            return list(all_matches.values())

        logger.info(f"    Starting from Round {current}")

        # Zbierz z obecnej rundy
        visible = self.driver.execute_script(JS_COLLECT_MATCHES) or []
        for m in visible:
            all_matches[m['url']] = m
        logger.info(f"    Round {current}: collected (total: {len(all_matches)})")

        # Klikaj wstecz do Round 1
        last_round = current
        same_count = 0
        for step in range(MAX_ROUNDS):
            if last_round <= 1:
                break

            click_result = self.driver.execute_script(JS_CLICK_PREV_ROUND)
            if click_result == 'disabled':
                logger.info(f"    Prev button disabled, stop")
                break
            if click_result != 'ok':
                logger.warning(f"    Click result: {click_result}")
                break

            time.sleep(ACTION_WAIT)

            new_round = self.driver.execute_script(JS_GET_CURRENT_ROUND)
            if new_round is None:
                break
            if new_round == last_round:
                same_count += 1
                if same_count >= 2:
                    logger.info(f"    Round didn't change, stop at {last_round}")
                    break
                continue
            same_count = 0
            last_round = new_round

            # Zbierz mecze z tej rundy
            visible = self.driver.execute_script(JS_COLLECT_MATCHES) or []
            added = 0
            for m in visible:
                if m['url'] not in all_matches:
                    all_matches[m['url']] = m
                    added += 1
            logger.info(f"    Round {new_round}: +{added} new (total: {len(all_matches)})")

        logger.info(f"    Total unique matches: {len(all_matches)}")
        return list(all_matches.values())

    def close(self):
        try:
            self.driver.quit()
        except:
            pass

# ==================== MAIN ====================

class Scraper:
    def __init__(self):
        self.browser = SofascoreScraper()
        self.cp = Checkpoint(CHECKPOINT_FILE)
        MATCHES_DIR.mkdir(parents=True, exist_ok=True)
        self.total = 0

    def save(self, key, code, season, matches):
        if not matches: return 0
        path = MATCHES_DIR / f"{key}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "tournament": code, "season": season,
                "count": len(matches),
                "saved_at": datetime.now().isoformat(),
                "matches": matches
            }, f, indent=2, ensure_ascii=False)
        return len(matches)

    def run(self):
        total_t = len(TOURNAMENTS)
        t_idx = 0
        logger.info("=" * 70)
        logger.info(f"Sofascore | {total_t} tournaments | {len(SEASONS_TO_SCRAPE)} seasons")
        logger.info("=" * 70)

        try:
            for code, info in TOURNAMENTS.items():
                t_idx += 1
                logger.info(f"\n[{t_idx}/{total_t}] {info['name']} ({code})")
                logger.info("-" * 70)
                t_total = 0

                for season in SEASONS_TO_SCRAPE:
                    key = f"{code}_{season.replace('/','-')}"

                    if self.cp.is_done(key):
                        logger.info(f"  {season}: SKIP (done)")
                        continue

                    logger.info(f"  {season}:")
                    try:
                        matches = self.browser.get_season_matches(info['url'], season)
                        if matches:
                            saved = self.save(key, code, season, matches)
                            self.total += saved
                            t_total += saved
                            self.cp.mark_done(key, saved)
                            logger.info(f"  {season}: SAVED {saved} matches (total: {self.total})")
                        else:
                            logger.info(f"  {season}: no matches")
                            self.cp.mark_done(key, 0)
                    except Exception as e:
                        logger.error(f"  {season}: ERROR - {e}")

                    time.sleep(1.5)

                logger.info(f"  >> {info['name']}: {t_total} matches")

        finally:
            self.browser.close()

        logger.info("\n" + "=" * 70)
        logger.info(f"DONE! Total: {self.total} matches in {MATCHES_DIR.resolve()}")
        logger.info("=" * 70)


def main():
    Scraper().run()

if __name__ == "__main__":
    main()