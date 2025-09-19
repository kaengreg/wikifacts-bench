import os
import re
import json
import requests
from typing import Any

from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, unquote

BASE_URL = 'https://nl.wikipedia.org'
MAIN_URL = BASE_URL + '/wiki/Wikipedia:Wist_je_dat'

OUTPUT_DIR = 'data/nl'


def get_day_links_from_archive(main_page_url: str) -> dict[str, dict[str, Any]]:
    """
    Get the year links from the main archive page.
    Current year link is the main page.
    """
    resp = requests.get(main_page_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    div = soup.find('div', class_='mw-heading mw-heading2 ext-discussiontools-init-section')
    if not div:
        raise ValueError("Could not find the div.")
    
    table = div.find_next('table', class_='wikitable')
    if not table:
        raise ValueError("Could not find the table.")

    archive_links = {}
    for tr in table.find_all('tr'):
        tds = tr.find_all('td')

        month = tds[0].find('a', href=True).text.strip()

        for a_tag in tds[1].find_all('a', href=True):        
            day = a_tag.text.strip()
            href = a_tag['href']
            full_url = urljoin(BASE_URL, href)
            exists = 'new' not in a_tag.get('class', [])

            full_day_name = day + ' ' + month
            archive_links[full_day_name] = {
                'url': full_url,
                'exists': exists
            }

    return archive_links


def _extract_fact_data(element: Tag, base_url: str) -> dict[str, Any]:
    """Extracts text, links, and relevant links from a BeautifulSoup Tag."""
    fact_text = "Wist je dat " + element.get_text(" ", strip=True).lstrip().lstrip('.').lstrip('…').lstrip()
    fact_text = fact_text.replace('  ', ' ')
    fact_text = fact_text.replace('\xa0', ' ')

    links = []
    for a in element.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("/wiki/"):
            continue

        full_url = unquote(urljoin(base_url, href))

        links.append(full_url)

    return {
        "text": fact_text,
        "links": links,
        "relevant_links": [],
    }


def parse_day_facts(day_url: str, day: str) -> list[dict]:
    """Parse the facts from the day page."""
    resp = requests.get(day_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    year_div = soup.find("div", class_="documentatie")
    year = None
    if year_div:
        year_p = year_div.find("p")
        if year_p:
            year_text = year_p.get_text(" ", strip=True)
            match = re.search(r'\d{4}', year_text)
            if match:
                year = match.group(0)

    results: list[dict] = []

    div = soup.find("div", class_="mw-content-ltr mw-parser-output")
    if not div:
        return []

    ul = div.find("ul")
    if not ul:
        return []

    for li in ul.find_all("li"):
        fact_data = _extract_fact_data(li, day_url)

        if fact_data['text']:
            results.append({
                "section": day + ' ' + year,
                **fact_data
            })

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    archive = get_day_links_from_archive(MAIN_URL)

    all_data: dict[str, dict[str, list[dict]]] = {}

    for day_name, data in archive.items():
        print(f"=== {day_name} ===")
        print(f" Parsing {day_name}:", end='')
        if data['exists']:
            facts = parse_day_facts(data['url'], day_name)
            print(f" {len(facts)} facts")

            _, month, year = tuple(facts[0]['section'].split(' '))
            if year not in all_data:
                all_data[year] = {}
            if month not in all_data[year]:
                all_data[year][month] = []
            all_data[year][month].extend(facts)
        else:
            print(" no page")

    # Sort the data by year, then write to file
    sorted_years = sorted(all_data.keys())
    sorted_all_data = {year: all_data[year] for year in sorted_years}

    out_path = os.path.join(OUTPUT_DIR, 'all_facts.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_all_data, f, ensure_ascii=False, indent=2)

    total_facts = sum(len(facts) for months in sorted_all_data.values() for facts in months.values())
    print(f"\nDone! All data written to {out_path}, total facts collected: {total_facts}.")


if __name__ == "__main__":
    main()
