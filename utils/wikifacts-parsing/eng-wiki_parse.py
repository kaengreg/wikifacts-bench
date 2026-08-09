import os
import re
import json
import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, unquote

BASE_URL = 'https://en.wikipedia.org'
MAIN_URL = BASE_URL + '/wiki/Wikipedia:Recent_additions'

OUTPUT_DIR = 'data/eng'

def get_month_links_from_archive(main_page_url):
    resp = requests.get(main_page_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    container = soup.find('div', class_='floatleft')
    if not container:
        raise ValueError("Could not find the container div with class 'floatleft'.")
    table = container.find('table')
    if not table: 
        raise ValueError("Could not find the table in the container div.")

    archive = {}
    for tr in container.find_all('tr'):
        th = tr.find('th')
        if not th:
            continue
        year = th.text.strip()
        if not re.match(r'^\d{4}$', year):
            continue
        
        months = []
        for td in tr.find_all('td'):
            a = td.find('a', href=True)
            if not a:
                continue
            month_name = a.text.strip()
            href = a['href']
            full_url = urljoin(BASE_URL, href)
            exists = 'new' not in a.get('class', [])
            months.append({
                'month': month_name,
                'url': full_url,
                'exists': exists
            })
        if months:
            archive[year] = months
    return archive    
    

def parse_month_facts(month_url: str) -> list[dict]:
    resp = requests.get(month_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results: list[dict] = []

    section_divs = soup.find_all("div", class_="mw-heading mw-heading3")
    for section_div in section_divs:
        header = section_div.find("h3")
        section_title = header.get_text(strip=True) if header else "(без заголовка)"

        for sib in section_div.find_next_siblings():
            if isinstance(sib, Tag) and "mw-heading3" in sib.get("class", []):
                break

            if not (isinstance(sib, Tag) and sib.name == "ul"):
                continue

            for li in sib.find_all("li")[1:]:
                fact_text = li.get_text(" ", strip=True)
                
                full_fact_text = "Did you know" + fact_text[3:]
                links = []
                relevant_links = []
                for a in li.find_all("a", href=True):
                    href = a["href"]
                    if not href.startswith("/wiki/"):
                        continue
                
                    full_url = unquote(urljoin(month_url, href))

                    if a.find_parent('b'):
                        relevant_links.append(full_url)
    
                    links.append(full_url)

                results.append({
                    "section": section_title,
                    "text": fact_text,
                    "links": links,
                    "relevant_links": relevant_links,
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

    out_path = os.path.join(OUTPUT_DIR, 'all_facts.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\nГотово! Все данные записаны в {out_path}, всего собрано {sum(len(facts) for months in all_data.values() for facts in months.values())} фактов.")


if __name__ == "__main__":
    main()
