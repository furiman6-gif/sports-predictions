"""
Dopisuje przyszle mecze z fixtures_22_ligues_2025_26.csv do wszystkie_sezony.csv kazdej ligi.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
FIXTURES_CSV = ROOT / "fixtures_22_ligues_2025_26.csv"
TODAY = pd.Timestamp.today().normalize()
SEZON = 2627

DIV_MAP = {
    "E0":  "England/Premier_League",
    "E1":  "England/Championship",
    "E2":  "England/League_One",
    "E3":  "England/League_Two",
    "EC":  "England/Conference",
    "D1":  "Germany/Bundesliga_1",
    "D2":  "Germany/Bundesliga_2",
    "SP1": "Spain/La_Liga",
    "SP2": "Spain/Segunda_Division",
    "F1":  "France/Ligue_1",
    "F2":  "France/Ligue_2",
    "I1":  "Italy/Serie_A",
    "I2":  "Italy/Serie_B",
    "B1":  "Belgium/First_Division_A",
    "N1":  "Netherlands/Eredivisie",
    "P1":  "Portugal/Primeira_Liga",
    "T1":  "Turkey/Super_Lig",
    "G1":  "Greece/Super_League",
    "SC0": "Scotland/Premiership",
    "SC1": "Scotland/Championship",
    "SC2": "Scotland/League_One",
    "SC3": "Scotland/League_Two",
}

TEAM_MAP = {
    # E0 Premier League
    "Brighton & Hove Albion": "Brighton",
    "Leeds United": "Leeds",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton": "Wolves",
    # E1 Championship
    "Birmingham City": "Birmingham",
    "Blackburn Rovers": "Blackburn",
    "Charlton Athletic": "Charlton",
    "Coventry City": "Coventry",
    "Derby County": "Derby",
    "Hull City": "Hull",
    "Norwich City": "Norwich",
    "Oxford United": "Oxford",
    "Preston North End": "Preston NE",
    "Queens Park Rangers": "QPR",
    "Sheffield Wednesday": "Sheffield Weds",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "West Bromwich Albion": "West Brom",
    "Ipswich Town": "Ipswich",
    "Middlesbrough": "Middlesbrough",
    "Millwall": "Millwall",
    "Leicester City": "Leicester City",
    "Portsmouth": "Portsmouth",
    "Sheffield Utd": "Sheffield Utd",
    "Watford": "Watford",
    "Wrexham": "Wrexham",
    # E2 League One
    "Bolton Wanderers": "Bolton",
    "Bradford City": "Bradford",
    "Burton Albion": "Burton",
    "Cardiff City": "Cardiff",
    "Doncaster Rovers": "Doncaster",
    "Exeter City": "Exeter",
    "Huddersfield Town": "Huddersfield",
    "Lincoln City": "Lincoln",
    "Luton Town": "Luton",
    "Mansfield Town": "Mansfield",
    "Northampton Town": "Northampton",
    "Peterborough United": "Peterboro",
    "Plymouth Argyle": "Plymouth",
    "Rotherham United": "Rotherham",
    "Stockport County": "Stockport",
    "Wigan Athletic": "Wigan",
    "Wycombe Wanderers": "Wycombe",
    # E3 League Two
    "Accrington Stanley": "Accrington",
    "Barrow AFC": "Barrow",
    "Bristol Rovers": "Bristol Rvs",
    "Cambridge United": "Cambridge",
    "Cheltenham Town": "Cheltenham",
    "Colchester United": "Colchester",
    "Crewe Alexandra": "Crewe",
    "Grimsby Town": "Grimsby",
    "Harrogate Town": "Harrogate",
    "Oldham Athletic": "Oldham",
    "Salford City": "Salford",
    "Shrewsbury Town": "Shrewsbury",
    "Swindon Town": "Swindon",
    "Tranmere Rovers": "Tranmere",
    # EC Conference
    "Aldershot Town": "Aldershot",
    "Altrincham FC": "Altrincham",
    "Boston United": "Boston Utd",
    "Carlisle United": "Carlisle",
    "FC Halifax Town": "Halifax",
    "Forest Green Rovers": "Forest Green",
    "Hartlepool United": "Hartlepool",
    "Scunthorpe United": "Scunthorpe",
    "Solihull Moors": "Solihull",
    "Southend United": "Southend",
    "Sutton United": "Sutton",
    "Truro City": "Truro",
    "Yeovil Town": "Yeovil",
    "York City": "York",
    # D1 Bundesliga
    "1. FC Heidenheim": "Heidenheim",
    "1. FC K\u00f6ln": "FC Koln",
    "1. FC Union Berlin": "Union Berlin",
    "1. FSV Mainz 05": "Mainz",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia M'gladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Augsburg": "Augsburg",
    "FC Bayern M\u00fcnchen": "Bayern Munich",
    "FC St. Pauli": "St Pauli",
    "Hamburger SV": "Hamburg",
    "SC Freiburg": "Freiburg",
    "SV Werder Bremen": "Werder Bremen",
    "TSG Hoffenheim": "Hoffenheim",
    "VfB Stuttgart": "Stuttgart",
    "VfL Wolfsburg": "Wolfsburg",
    "RasenBallsport Leipzig": "RB Leipzig",
    "RB Leipzig": "RB Leipzig",
    # D2 Bundesliga 2
    "1. FC Kaiserslautern": "Kaiserslautern",
    "1. FC Magdeburg": "Magdeburg",
    "1. FC N\u00fcrnberg": "Nurnberg",
    "Arminia Bielefeld": "Bielefeld",
    "Darmstadt 98": "Darmstadt",
    "Eintracht Braunschweig": "Braunschweig",
    "FC Schalke 04": "Schalke 04",
    "Fortuna D\u00fcsseldorf": "Fortuna Dusseldorf",
    "Hannover 96": "Hannover",
    "Hertha BSC": "Hertha",
    "Karlsruher SC": "Karlsruhe",
    "SC Paderborn 07": "Paderborn",
    "SG Dynamo Dresden": "Dresden",
    "SV 07 Elversberg": "Elversberg",
    "SpVgg Greuther F\u00fcrth": "Greuther Furth",
    "VfL Bochum 1848": "Bochum",
    # SP1 La Liga
    "Athletic Club": "Ath Bilbao",
    "Atl\u00e9tico Madrid": "Ath Madrid",
    "Celta Vigo": "Celta",
    "Deportivo Alav\u00e9s": "Alaves",
    "Espanyol": "Espanol",
    "FC Barcelona": "Barcelona",
    "Girona FC": "Girona",
    "Levante UD": "Levante",
    "Rayo Vallecano": "Vallecano",
    "Real Betis": "Betis",
    "Real Oviedo": "Oviedo",
    "Real Sociedad": "Sociedad",
    "Real Valladolid": "Valladolid",
    "SD Huesca": "Huesca",
    "Sporting Gij\u00f3n": "Sp Gijon",
    # SP2 Segunda
    "AD Ceuta": "Ceuta",
    "Albacete Balompi\u00e9": "Albacete",
    "Almer\u00eda": "Almeria",
    "Burgos Club de F\u00fatbol": "Burgos",
    "CD castell\u00f3n": "Castellon",
    "CD Castell\u00f3n": "Castellon",
    "C\u00e1diz": "Cadiz",
    "C\u00f3rdoba": "Cordoba",
    "Elche": "Elche",
    "FC Cartagena": "Cartagena",
    "Gimnàstic": "Gimnastic",
    "Granada CF": "Granada",
    "Huesca": "Huesca",
    "Levante UD": "Levante",
    "Mirand\u00e9s": "Mirandes",
    "Racing de Ferrol": "Racing Ferrol",
    "Racing de Santander": "Racing Santander",
    "Real Oviedo": "Oviedo",
    "Real Racing Club": "Racing Santander",
    "Real Valladolid": "Valladolid",
    "Real Zaragoza": "Zaragoza",
    "SD Eibar": "Eibar",
    "Villarreal CF B": "Villarreal B",
    # F1 Ligue 1
    "AS Monaco": "Monaco",
    "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille",
    "Paris Saint-Germain": "Paris SG",
    "RC Lens": "Lens",
    "RC Strasbourg": "Strasbourg",
    "Stade Brestois": "Brest",
    "Stade Rennais": "Rennes",
    "GFC Ajaccio": "Ajaccio GFCO",
    "SC Bastia": "Bastia",
    "Saint-\u00c9tienne": "St Etienne",
    "Clermont Foot": "Clermont",
    # F2 Ligue 2
    "Amiens SC": "Amiens",
    "Annecy FC": "Annecy",
    "Grenoble Foot 38": "Grenoble",
    "Red Star FC": "Red Star",
    "Rodez AF": "Rodez",
    "Saint-\u00c9tienne": "St Etienne",
    "Stade Lavallois": "Laval",
    "Stade de Reims": "Reims",
    "US Boulogne C\u00f4te-d'Opale": "Boulogne",
    "USL Dunkerque": "Dunkerque",
    # I1 Serie A
    "Hellas Verona": "Verona",
    "Parma Calcio 1913": "Parma",
    "SPAL 2013": "Spal",
    "AC Milan": "Milan",
    # I2 Serie B
    "S\u00fcdtirol": "Sudtirol",
    "US Avellino 1912": "Avellino",
    # B1 Belgium
    "Club Brugge KV": "Club Brugge",
    "FCV Dender": "Dender",
    "KAA Gent": "Gent",
    "KRC Genk": "Genk",
    "KV Mechelen": "Mechelen",
    "KVC Westerlo": "Westerlo",
    "Beerschot VA": "Beerschot",
    "RWDM Brussels FC": "RWDM",
    "Anderlecht": "Anderlecht",
    "R. Charleroi SC": "Charleroi",
    "OH Leuven": "OH Leuven",
    "KV Kortrijk": "Kortrijk",
    "KV Oostende": "Oostende",
    "Cercle Brugge": "Cercle Brugge",
    "RFC Seraing": "Seraing",
    "Zulte Waregem": "Waregem",
    "Waasland-Beveren": "Waasland-Beveren",
    "St. Truiden VV": "St Truiden",
    "Royale Union SG": "St. Gilloise",
    # N1 Eredivisie
    "AFC Ajax": "Ajax",
    "FC Groningen": "Groningen",
    "FC Twente": "Twente",
    "FC Utrecht": "Utrecht",
    "FC Volendam": "Volendam",
    "Fortuna Sittard": "For Sittard",
    "Heracles Almelo": "Heracles",
    "NEC Nijmegen": "Nijmegen",
    "PEC Zwolle": "Zwolle",
    "SC Heerenveen": "Heerenveen",
    "SC Telstar": "Telstar",
    "Feyenoord": "Feyenoord",
    "PSV Eindhoven": "PSV",
    "AZ Alkmaar": "AZ",
    "Vitesse": "Vitesse",
    "Go Ahead Eagles": "Go Ahead",
    "RKC Waalwijk": "RKC Waalwijk",
    "Sparta Rotterdam": "Sparta",
    "Willem II": "Willem II",
    "Almere City FC": "Almere City",
    # P1 Primeira Liga
    "AVS - Futebol SAD": "AVS",
    "CD Nacional": "Nacional",
    "CF Estrela Amadora": "Estrela",
    "Estoril Praia": "Estoril",
    "FC Alverca": "Alverca",
    "FC Arouca": "Arouca",
    "FC Porto": "Porto",
    "SL Benfica": "Benfica",
    "Sporting CP": "Sp Lisbon",
    "SC Braga": "Braga",
    "Vitória SC": "Vitoria SC",
    "Moreirense FC": "Moreirense",
    "Casa Pia AC": "Casa Pia",
    "Gil Vicente FC": "Gil Vicente",
    "GD Chaves": "Chaves",
    "Rio Ave FC": "Rio Ave",
    "Santa Clara": "Santa Clara",
    "Famalic\u00e3o": "Famalicao",
    "Boavista FC": "Boavista",
    # T1 Super Lig
    "Ba\u015fak\u015fehir FK": "Buyuksehyr",
    "Be\u015fikta\u015f JK": "Besiktas",
    "Ey\u00fcpspor": "Eyupspor",
    "Fatih Karag\u00fcmr\u00fck": "Karagumruk",
    "Fenerbah\u00e7e": "Fenerbahce",
    "Gaziantep FK": "Gaziantep",
    "Alanyaspor": "Alanyaspor",
    "Antalyaspor": "Antalyaspor",
    "Beşiktaş JK": "Besiktas",
    "Galatasaray": "Galatasaray",
    "Trabzonspor": "Trabzonspor",
    "Sivasspor": "Sivasspor",
    "Kasımpaşa": "Kasimpasa",
    "Hatayspor": "Hatayspor",
    "Konyaspor": "Konyaspor",
    "Kayserispor": "Kayserispor",
    "Ankaragücü": "Ankaragucu",
    "Rizespor": "Rizespor",
    "Pendikspor": "Pendikspor",
    "Samsunspor": "Samsunspor",
    "İstanbul Başakşehir": "Buyuksehyr",
    # G1 Super League Greece
    "AE Kifisia": "Kifisia",
    "AEK Athens": "AEK",
    "AEL Novibet": "AEL",
    "APO Levadiakos": "Levadeiakos",
    "APS Atromitos Athinon": "Atromitos",
    "Aris Thessaloniki": "Aris",
    "Asteras Aktor": "Asteras Tripolis",
    "GFS Panetolikos": "Panetolikos",
    "MGS Panserraikos": "Panserraikos",
    "NPS Volos": "Volos NFC",
    "Olympiacos FC": "Olympiakos",
    "Panathinaikos FC": "Panathinaikos",
    "PAOK FC": "PAOK",
    "OFI Crete": "OFI Crete",
    "Ionikos": "Ionikos",
    "PAS Giannina": "PAS Giannina",
    # SC0 Premiership
    "Dundee FC": "Dundee",
    "Falkirk FC": "Falkirk",
    "Heart of Midlothian": "Hearts",
    "St. Mirren": "St Mirren",
    "Celtic FC": "Celtic",
    "Rangers FC": "Rangers",
    "Hibernian FC": "Hibernian",
    "Aberdeen FC": "Aberdeen",
    "Motherwell FC": "Motherwell",
    "Ross County": "Ross County",
    "Kilmarnock FC": "Kilmarnock",
    "St. Johnstone": "St Johnstone",
    # SC1 Championship
    "Airdrieonians": "Airdrie Utd",
    "Ayr United": "Ayr",
    "Dunfermline Athletic": "Dunfermline",
    "Greenock Morton": "Morton",
    "Partick Thistle": "Partick",
    "Queen's Park FC": "Queens Park",
    "Raith Rovers": "Raith Rvs",
    # SC2 League One
    "Alloa Athletic": "Alloa",
    "Hamilton Academical": "Hamilton",
    "Inverness Caledonian Thistle": "Inverness C",
    "Kelty Hearts F.C.": "Kelty Hearts",
    "Queen of The South": "Queen of Sth",
    # SC3 League Two
    "Clyde FC": "Clyde",
    "Edinburgh City F.C.": "Edinburgh City",
    "Elgin City": "Elgin",
    "Forfar Athletic": "Forfar",
    "Stirling Albion": "Stirling",
    "The Spartans FC": "Spartans",
}


def map_team(name: str) -> str:
    return TEAM_MAP.get(name, name)


def main():
    fix = pd.read_csv(FIXTURES_CSV, encoding="utf-8-sig")
    fix["Date"] = pd.to_datetime(fix["Date"], dayfirst=True, errors="coerce")
    fix = fix[fix["Date"] >= TODAY]

    total_added = 0

    for div, rel_path in DIV_MAP.items():
        csv_path = ROOT / rel_path / "wszystkie_sezony.csv"
        if not csv_path.exists():
            print(f"  BRAK: {csv_path}")
            continue

        div_fix = fix[fix["Div"] == div].copy()
        if div_fix.empty:
            continue

        div_fix["HomeTeam"] = div_fix["HomeTeam"].map(map_team)
        div_fix["AwayTeam"] = div_fix["AwayTeam"].map(map_team)
        div_fix["Date"] = div_fix["Date"].dt.strftime("%d/%m/%Y")

        df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8-sig", on_bad_lines="warn")

        # klucz unikalnosci
        existing = set(
            zip(df["Date"].astype(str), df["HomeTeam"].astype(str), df["AwayTeam"].astype(str))
        )

        new_rows = []
        for _, row in div_fix.iterrows():
            key = (row["Date"], row["HomeTeam"], row["AwayTeam"])
            if key not in existing:
                new_rows.append({
                    "Sezon": SEZON,
                    "Div": div,
                    "Date": row["Date"],
                    "HomeTeam": row["HomeTeam"],
                    "AwayTeam": row["AwayTeam"],
                })

        if not new_rows:
            print(f"  {div}: wszystkie juz sa")
            continue

        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([df, new_df], ignore_index=True)
        combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  {div} ({rel_path}): +{len(new_rows)} nowych meczow")
        total_added += len(new_rows)

    print(f"\nGotowe! Lacznie dopisano: {total_added} meczow")


if __name__ == "__main__":
    main()
