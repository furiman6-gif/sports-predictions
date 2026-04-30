import sys
sys.path.insert(0, '.')
from charting_match_level import parse_match_page

html_path = r"C:\Users\furim\AppData\Local\Temp\tmpxbqrutdb.html"
with open(html_path, encoding="utf-8") as f:
    html = f.read()

result = parse_match_page(html, "Matteo Berrettini", "Daniil Medvedev")
print("Berrettini:", result["Matteo Berrettini"])
print("Medvedev:  ", result["Daniil Medvedev"])
