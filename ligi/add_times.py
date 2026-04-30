import pandas as pd, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

TEAM_MAP = {
    'Brighton & Hove Albion':'Brighton','Leeds United':'Leeds','Manchester City':'Man City',
    'Manchester United':'Man United','Newcastle United':'Newcastle','Nottingham Forest':"Nott'm Forest",
    'Tottenham Hotspur':'Tottenham','West Ham United':'West Ham','Wolverhampton':'Wolves',
    'Birmingham City':'Birmingham','Blackburn Rovers':'Blackburn','Charlton Athletic':'Charlton',
    'Coventry City':'Coventry','Derby County':'Derby','Hull City':'Hull','Norwich City':'Norwich',
    'Oxford United':'Oxford','Preston North End':'Preston NE','Queens Park Rangers':'QPR',
    'Sheffield Wednesday':'Sheffield Weds','Stoke City':'Stoke','Swansea City':'Swansea',
    'West Bromwich Albion':'West Brom','Bolton Wanderers':'Bolton','Bradford City':'Bradford',
    'Burton Albion':'Burton','Cardiff City':'Cardiff','Doncaster Rovers':'Doncaster',
    'Exeter City':'Exeter','Huddersfield Town':'Huddersfield','Lincoln City':'Lincoln',
    'Luton Town':'Luton','Mansfield Town':'Mansfield','Northampton Town':'Northampton',
    'Peterborough United':'Peterboro','Plymouth Argyle':'Plymouth','Rotherham United':'Rotherham',
    'Stockport County':'Stockport','Wigan Athletic':'Wigan','Wycombe Wanderers':'Wycombe',
    'Accrington Stanley':'Accrington','Barrow AFC':'Barrow','Bristol Rovers':'Bristol Rvs',
    'Cambridge United':'Cambridge','Cheltenham Town':'Cheltenham','Colchester United':'Colchester',
    'Crewe Alexandra':'Crewe','Grimsby Town':'Grimsby','Harrogate Town':'Harrogate',
    'Oldham Athletic':'Oldham','Salford City':'Salford','Shrewsbury Town':'Shrewsbury',
    'Swindon Town':'Swindon','Tranmere Rovers':'Tranmere',
    'Aldershot Town':'Aldershot','Altrincham FC':'Altrincham','Boston United':'Boston Utd',
    'Carlisle United':'Carlisle','FC Halifax Town':'Halifax','Forest Green Rovers':'Forest Green',
    'Hartlepool United':'Hartlepool','Scunthorpe United':'Scunthorpe','Solihull Moors':'Solihull',
    'Southend United':'Southend','Sutton United':'Sutton','Truro City':'Truro',
    'Yeovil Town':'Yeovil','York City':'York',
    '1. FC Heidenheim':'Heidenheim','1. FC K\u00f6ln':'FC Koln','1. FC Union Berlin':'Union Berlin',
    '1. FSV Mainz 05':'Mainz','Bayer 04 Leverkusen':'Leverkusen','Borussia Dortmund':'Dortmund',
    "Borussia M'gladbach":"M'gladbach",'Eintracht Frankfurt':'Ein Frankfurt','FC Augsburg':'Augsburg',
    'FC Bayern M\u00fcnchen':'Bayern Munich','FC St. Pauli':'St Pauli','Hamburger SV':'Hamburg',
    'SC Freiburg':'Freiburg','SV Werder Bremen':'Werder Bremen','TSG Hoffenheim':'Hoffenheim',
    'VfB Stuttgart':'Stuttgart','VfL Wolfsburg':'Wolfsburg','RB Leipzig':'RB Leipzig',
    '1. FC Kaiserslautern':'Kaiserslautern','1. FC Magdeburg':'Magdeburg','1. FC N\u00fcrnberg':'Nurnberg',
    'Arminia Bielefeld':'Bielefeld','Darmstadt 98':'Darmstadt','Eintracht Braunschweig':'Braunschweig',
    'FC Schalke 04':'Schalke 04','Fortuna D\u00fcsseldorf':'Fortuna Dusseldorf','Hannover 96':'Hannover',
    'Hertha BSC':'Hertha','Karlsruher SC':'Karlsruhe','SC Paderborn 07':'Paderborn',
    'SG Dynamo Dresden':'Dresden','SV 07 Elversberg':'Elversberg','SpVgg Greuther F\u00fcrth':'Greuther Furth',
    'VfL Bochum 1848':'Bochum','Athletic Club':'Ath Bilbao','Atl\u00e9tico Madrid':'Ath Madrid',
    'Celta Vigo':'Celta','Deportivo Alav\u00e9s':'Alaves','Espanyol':'Espanol','FC Barcelona':'Barcelona',
    'Girona FC':'Girona','Levante UD':'Levante','Rayo Vallecano':'Vallecano','Real Betis':'Betis',
    'Real Oviedo':'Oviedo','Real Sociedad':'Sociedad','Real Valladolid':'Valladolid','SD Huesca':'Huesca',
    'AS Monaco':'Monaco','Olympique Lyonnais':'Lyon','Olympique de Marseille':'Marseille',
    'Paris Saint-Germain':'Paris SG','RC Lens':'Lens','RC Strasbourg':'Strasbourg',
    'Stade Brestois':'Brest','Stade Rennais':'Rennes','GFC Ajaccio':'Ajaccio GFCO',
    'Saint-\u00c9tienne':'St Etienne','Clermont Foot':'Clermont','Amiens SC':'Amiens',
    'Annecy FC':'Annecy','Grenoble Foot 38':'Grenoble','Red Star FC':'Red Star','Rodez AF':'Rodez',
    'Stade Lavallois':'Laval','Stade de Reims':'Reims',
    'USL Dunkerque':'Dunkerque','Hellas Verona':'Verona','AC Milan':'Milan',
    'S\u00fcdtirol':'Sudtirol','Club Brugge KV':'Club Brugge','FCV Dender':'Dender',
    'KAA Gent':'Gent','KRC Genk':'Genk','KV Mechelen':'Mechelen','KVC Westerlo':'Westerlo',
    'AFC Ajax':'Ajax','FC Groningen':'Groningen','FC Twente':'Twente','FC Utrecht':'Utrecht',
    'FC Volendam':'Volendam','Fortuna Sittard':'For Sittard','Heracles Almelo':'Heracles',
    'NEC Nijmegen':'Nijmegen','PEC Zwolle':'Zwolle','SC Heerenveen':'Heerenveen','SC Telstar':'Telstar',
    'PSV Eindhoven':'PSV','AZ Alkmaar':'AZ',
    'AVS - Futebol SAD':'AVS','CD Nacional':'Nacional','CF Estrela Amadora':'Estrela',
    'Estoril Praia':'Estoril','FC Alverca':'Alverca','FC Arouca':'Arouca','FC Porto':'Porto',
    'SL Benfica':'Benfica','Sporting CP':'Sp Lisbon','SC Braga':'Braga',
    'Moreirense FC':'Moreirense','Casa Pia AC':'Casa Pia','Gil Vicente FC':'Gil Vicente',
    'GD Chaves':'Chaves','Rio Ave FC':'Rio Ave','Boavista FC':'Boavista',
    'Ba\u015fak\u015fehir FK':'Buyuksehyr','Be\u015fikta\u015f JK':'Besiktas','Ey\u00fcpspor':'Eyupspor',
    'Fatih Karag\u00fcmr\u00fck':'Karagumruk','Fenerbah\u00e7e':'Fenerbahce','Gaziantep FK':'Gaziantep',
    'AE Kifisia':'Kifisia','AEK Athens':'AEK','AEL Novibet':'AEL','APO Levadiakos':'Levadeiakos',
    'APS Atromitos Athinon':'Atromitos','Aris Thessaloniki':'Aris','Asteras Aktor':'Asteras Tripolis',
    'GFS Panetolikos':'Panetolikos','MGS Panserraikos':'Panserraikos','NPS Volos':'Volos NFC',
    'Olympiacos FC':'Olympiakos','Panathinaikos FC':'Panathinaikos',
    'Dundee FC':'Dundee','Falkirk FC':'Falkirk','Heart of Midlothian':'Hearts','St. Mirren':'St Mirren',
    'Celtic FC':'Celtic','Rangers FC':'Rangers','Hibernian FC':'Hibernian','Aberdeen FC':'Aberdeen',
    'Motherwell FC':'Motherwell','St. Johnstone':'St Johnstone',
    'Airdrieonians':'Airdrie Utd','Ayr United':'Ayr','Dunfermline Athletic':'Dunfermline',
    'Greenock Morton':'Morton','Partick Thistle':'Partick',"Queen's Park FC":'Queens Park',
    'Raith Rovers':'Raith Rvs','Alloa Athletic':'Alloa','Hamilton Academical':'Hamilton',
    'Inverness Caledonian Thistle':'Inverness C','Kelty Hearts F.C.':'Kelty Hearts',
    'Queen of The South':'Queen of Sth','Clyde FC':'Clyde','Edinburgh City F.C.':'Edinburgh City',
    'Elgin City':'Elgin','Forfar Athletic':'Forfar','Stirling Albion':'Stirling',
    'The Spartans FC':'Spartans',
}

ROOT = 'C:/Users/furim/Desktop/FINAL_tenis/ligi'
fix = pd.read_csv(f'{ROOT}/fixtures_22_ligues_2025_26.csv', encoding='utf-8-sig')
fix['HomeTeam_m'] = fix['HomeTeam'].map(lambda x: TEAM_MAP.get(x, x))
fix['AwayTeam_m'] = fix['AwayTeam'].map(lambda x: TEAM_MAP.get(x, x))
fix['Date_norm'] = pd.to_datetime(fix['Date'], dayfirst=True, errors='coerce').dt.strftime('%d/%m/%Y')
fix_lookup = fix[['Date_norm','HomeTeam_m','AwayTeam_m','Time']].rename(
    columns={'Date_norm':'Date','HomeTeam_m':'HomeTeam','AwayTeam_m':'AwayTeam'})

pred = pd.read_csv(f'{ROOT}/auto_outputs_future/future_predictions_1row_all_targets.csv', low_memory=False, encoding='utf-8-sig')
pred['Date_str'] = pd.to_datetime(pred['Date'], dayfirst=True, errors='coerce').dt.strftime('%d/%m/%Y')
merged = pred.merge(fix_lookup, left_on=['Date_str','HomeTeam','AwayTeam'],
                    right_on=['Date','HomeTeam','AwayTeam'], how='left', suffixes=('','_fix'))
merged = merged.drop(columns=['Date_fix','Date_str'], errors='ignore')

cols = merged.columns.tolist()
if 'Time' in cols:
    cols.remove('Time')
date_idx = cols.index('Date')
cols.insert(date_idx+1, 'Time')
merged = merged[cols]

out = f'{ROOT}/auto_outputs_future/future_predictions_1row_all_targets.csv'
merged.to_csv(out, index=False, encoding='utf-8-sig')

matched = merged['Time'].notna().sum()
print(f'Dodano godziny: {matched}/{len(merged)} meczow')
print(merged[merged['Time'].notna()][['Date','Time','HomeTeam','AwayTeam']].head(12).to_string(index=False))
