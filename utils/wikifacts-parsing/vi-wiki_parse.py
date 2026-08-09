import os
import re
import json
import requests
from typing import Any
from collections import defaultdict

from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, unquote

BASE_URL = 'https://vi.wikipedia.org'
MAIN_URL = BASE_URL + '/wiki/Wikipedia:B%E1%BA%A1n_c%C3%B3_bi%E1%BA%BFt'

OUTPUT_DIR = 'data/vi'

NUMBER_TO_MONTH = {
    1: 'Tháng một',
    2: 'Tháng hai',
    3: 'Tháng ba',
    4: 'Tháng tư',
    5: 'Tháng năm',
    6: 'Tháng sáu',
    7: 'Tháng bảy',
    8: 'Tháng tám',
    9: 'Tháng chín',
    10: 'Tháng mười',
    11: 'Tháng mười một',
    12: 'Tháng mười hai',
}


def calculate_month_number_from_week_number(week_number: int) -> int:
    """
    Calculate the approximate month number from the week number.
    """
    month_number = (week_number - 1) * 7 // 30 + 1
    if month_number > 12:
        month_number = 12
    
    return month_number


def get_year_links_from_archive(main_page_url: str) -> dict[str, dict[str, Any]]:
    """
    Get the year links from the main archive page.
    Current year link is the main page.
    """
    resp = requests.get(main_page_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    div = soup.find('div', class_='hlist')
    if not div:
        raise ValueError("Could not find the div.")

    archive_links = {}
    for a_tag in div.find_all('a', href=True):        
        year = a_tag.text.strip()
        if re.match(r'^\d{4}$', year):
            href = a_tag['href']
            full_url = urljoin(BASE_URL, href)
            exists = 'new' not in a_tag.get('class', [])
            archive_links[year] = {
                'url': full_url,
                'exists': exists
            }

    return archive_links


def _extract_fact_data(element: Tag, base_url: str) -> dict[str, Any]:
    """Extracts text, links, and relevant links from a BeautifulSoup Tag."""
    fact_text = "Bạn có biết " + element.get_text(" ", strip=True).lstrip().lstrip('.').lstrip('…').lstrip()
    fact_text = fact_text.replace('\xa0', ' ').replace('  ', ' ')

    links = []
    relevant_links = []
    for a in element.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("/wiki/"):
            continue

        full_url = unquote(urljoin(base_url, href))

        if a.find_parent('b'):
            relevant_links.append(full_url)

        links.append(full_url)

    return {
        "text": fact_text,
        "links": links,
        "relevant_links": relevant_links,
    }


def parse_year_facts(year_url: str) -> list[dict]:
    """Parse the facts from the year page."""
    resp = requests.get(year_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results: list[dict] = []

    # Parse facts from all uls in section div
    div = soup.find("div", class_="mw-content-ltr mw-parser-output")
    if not div:
        raise ValueError("Could not find the div.")

    table = div.find('table')
    if not table:
        raise ValueError("Could not find the table.")

    for tr in table.find_all('tr'):
        for td in tr.find_all('td'):
            try:
                section = td.find('h3').text.strip()
            except Exception:
                continue

            ul = td.find('ul')
            for li in ul.find_all('li'):
                fact_data = _extract_fact_data(li, year_url)

                if fact_data['text']:
                    results.append({
                        "section": section,
                        **fact_data
                    })
    
    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    archive = get_year_links_from_archive(MAIN_URL)

    all_data: dict[str, dict[str, list[dict]]] = {}

    for year, data in sorted(archive.items()):
        print(f"=== {year} ===")
        print(f" Parsing {year}:", end='')
        if data['exists']:
            facts = parse_year_facts(data['url'])
            print(f" {len(facts)} facts")

            year_data: dict[str, list[dict]] = defaultdict(list)
            for fact in facts:
                month_name = NUMBER_TO_MONTH[calculate_month_number_from_week_number(int(fact['section'].split()[1]))]
                year_data[month_name].append(fact)
            all_data[year] = dict(year_data)
        else:
            facts = []
            print(" no page")
            all_data[year] = {}

    out_path = os.path.join(OUTPUT_DIR, 'all_facts.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    total_facts = sum(len(facts) for months in all_data.values() for facts in months.values())
    print(f"\nDone! All data written to {out_path}, total facts collected: {total_facts}.")


if __name__ == "__main__":
    main()
