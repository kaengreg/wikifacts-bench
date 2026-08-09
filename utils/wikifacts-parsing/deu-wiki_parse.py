import os
import re
import json
import requests
import unicodedata
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, unquote

BASE_URL = 'https://de.wikipedia.org'
MAIN_URL = BASE_URL + '/wiki/Wikipedia:Hauptseite/Schon_gewusst/Archiv'

OUTPUT_DIR = 'data/deu'

def get_month_links_from_archive(url):
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    content = soup.find('div', class_="mw-content-ltr mw-parser-output")
    
    tables = content.find_all('table')
    table = tables[1]
    archive = {}

    for tr in table.find_all('tr'):
        tds = tr.find_all('td')
        if not tds:
            continue
        
        year = tds[0].find('b').text
        months = []
        for td in tds[1:]:
            a = td.find('a')
            if not a:
                continue 
            months.append({
                "month": a.text.strip(),
                "url": urljoin(url, a['href']),
                'exists': 'new' not in a.get('class', [])
            })
        if months:
            archive[str(year)] = months
    return archive


def parse_month_facts(month_url: str) -> list[dict]:
    resp = requests.get(month_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    cells = soup.find_all("div", class_="hintergrundfarbe-basis")
    for cell in cells:
        date_span = cell.find("span", style=lambda s: s and "font-weight:bold" in s)
        date = date_span.text.strip() if date_span else None

        paras = cell.find("p")
        text = " ".join(p.get_text(" ", strip=True) for p in paras)

        links = []
        for a in paras.find_all("a", href=True):
            if a["href"].startswith("/wiki/"):
                links.append(unquote(urljoin(month_url, a["href"])))

        results.append({
            "section":  str(date),
            "text":  str(text),
            "links": links
        })

    return results

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    archive = get_month_links_from_archive(MAIN_URL)

    all_data: dict[str, dict[str, list[dict]] ] = {}

    for year, months in archive.items():
        print(f"=== {year} ===")
        year_data: dict[str, list[dict]] = {}
        for m in months:
            month_name = m['month']
            print(f" Парсим {year} — {month_name}:", end=' ')
            if m['exists']:
                facts = parse_month_facts(m['url'])
                print(f"{len(facts)} фактов")
            else:
                facts = []
                print("нет страницы")
            year_data[month_name] = facts
        all_data[year] = year_data

    print(all_data)
    out_path = os.path.join(OUTPUT_DIR, 'all_facts.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\nГотово! Все данные записаны в {out_path}, всего собрано {sum(len(facts) for months in all_data.values() for facts in months.values())} фактов.")


if __name__ == "__main__":
    main()
