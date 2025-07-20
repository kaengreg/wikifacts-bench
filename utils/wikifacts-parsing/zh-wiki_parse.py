import os
import re
import json
import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, unquote

BASE_URL = 'https://zh.wikipedia.org'
MAIN_URL = BASE_URL + '/wiki/Wikipedia:%E6%96%B0%E6%9D%A1%E7%9B%AE%E6%8E%A8%E8%8D%90'

OUTPUT_DIR = 'data/cn'


def get_month_links_from_archive(main_page_url):
    resp = requests.get(main_page_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    container = soup.find('div', class_='center')
    if not container:
        raise ValueError("Could not find the container div with class 'center'.")
    table = container.find('table', class_='ombox ombox-notice mbox-small')
    if not table: 
        raise ValueError("Could not find the table with class 'ombox ombox-notice mbox-small' in the container div.")
    main_row = table.find_all('tr')[1]
    if not main_row:
        raise ValueError("Could not find the main row in the table.")
    row_table = main_row.find('table')
    if not row_table:
        raise ValueError("Could not find the row table in the main row.")

    archive = {}
    for tr in row_table.find_all('tr'):
        th = tr.find('th')
        if not th:
            continue
        year = th.text.strip()[:-1]
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

    div = soup.find("div", class_="mw-heading mw-heading2 ext-discussiontools-init-section")
    if not div:
        raise ValueError("Could not find the div with class 'mw-heading mw-heading2 ext-discussiontools-init-section'.")
    div_siblings = div.find_next_siblings()
    section_title = div_siblings[0].get_text(strip=True)

    for elem in div_siblings[1:]:
        if not (isinstance(elem, Tag) and elem.name == "ul"):
            continue

        for li in elem.find_all("li")[:]:
            fact_text = li.get_text(" ", strip=True)
            
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
