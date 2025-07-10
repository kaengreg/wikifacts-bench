import os
import re
import json
import requests
from typing import Any
from collections import defaultdict
from copy import copy

from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, unquote

BASE_URL = 'https://fr.wikipedia.org'
MAIN_URL = BASE_URL + '/wiki/Wikip%C3%A9dia:Le_saviez-vous_%3F'
OUTPUT_DIR = 'data/fr'

FRENCH_MONTHS = [
    'Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin',
    'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre'
]
MONTH_NUM_TO_NAME = {i + 1: name for i, name in enumerate(FRENCH_MONTHS)}


def get_year_links_from_archive(main_page_url: str) -> dict[str, dict[str, Any]]:
    """Get the year links from the main archive page."""
    resp = requests.get(main_page_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    sibling_div = soup.find('div', class_='mw-heading mw-heading2 ext-discussiontools-init-section')
    if not sibling_div:
        raise ValueError("Could not find the nearest sibling div with class 'mw-heading mw-heading2 ext-discussiontools-init-section'.")
    table = sibling_div.find_next_siblings()[1]
    if not table:
        raise ValueError("Could not find the table.")
    p = table.find('p')
    if not p:
        raise ValueError("Could not find the paragraph.")
    
    archive_links = {}
    for a_tag in p.find_all('a', href=True)[1:]: # skip the modifier link
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
    # Create a copy of the element to safely remove unwanted tags for text extraction.
    text_element = copy(element)
    for tag_to_remove in text_element.find_all(['figure', 'dl']):
        tag_to_remove.decompose()
        
    fact_text = text_element.get_text(" ", strip=True).replace('\xa0', ' ')

    links = []
    relevant_links = []
    for a in element.find_all("a", href=True):
        if a.find_parent(['figure', 'dl']):
            continue

        href = a["href"]
        if not href.startswith("/wiki/"):
            continue

        full_url = unquote(urljoin(base_url, href))

        if a.find_parent('b'):
            relevant_links.append(full_url)

        links.append(full_url)

    dl_tag = element.find('dl')
    section_from_dl = None
    if dl_tag:
        dl_text = dl_tag.get_text()
        
        # Format 1: YYYY-MM-DD
        match_ymd = re.search(r'(\d{4})-(\d{2})-(\d{2})', dl_text)
        if match_ymd:
            y, m, d = match_ymd.groups()
            month_name = MONTH_NUM_TO_NAME.get(int(m))
            if month_name:
                section_from_dl = f"{int(d):02d} {month_name} {y}"
        
        # Format 2: DD month_name YYYY
        else:
            month_pattern = '|'.join(FRENCH_MONTHS)
            match_dmy = re.search(r'(\d{1,2})\s+(' + month_pattern + r')\s+(\d{4})', dl_text, re.IGNORECASE)
            if match_dmy:
                d, month_name, y = match_dmy.groups()
                month_name_capitalized = next((m for m in FRENCH_MONTHS if m.lower() == month_name.lower()), month_name)
                section_from_dl = f"{int(d):02d} {month_name_capitalized} {y}"

    return {
        "text": fact_text,
        "links": links,
        "relevant_links": relevant_links,
        "section_from_dl": section_from_dl,
    }


def parse_year_facts(year_url: str, year: str) -> list[dict]:
    """Parse the facts from the year page."""
    resp = requests.get(year_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results: list[dict] = []

    # Parse facts from all uls in section div
    div = soup.find("div", class_="mw-content-ltr mw-parser-output")
    uls = div.find_all("ul")

    for ul in uls:
        for li in ul.find_all("li"):
            fact_data = _extract_fact_data(li, year_url)
            section = fact_data.pop('section_from_dl', None) or year
            if fact_data['text']:
                results.append({
                    "section": section,
                    **fact_data
                })

    return results


def _extract_year_and_month_from_section(section: str) -> tuple[str, str]:
    """Extracts year and month from the section string."""
    month_pattern = '|'.join(FRENCH_MONTHS)
    # Match "DD Month YYYY"
    match_dmy = re.match(r'\d{1,2}\s+(' + month_pattern + r')\s+(\d{4})', section, re.IGNORECASE)
    if match_dmy:
        month, year = match_dmy.groups()
        month_capitalized = next((m for m in FRENCH_MONTHS if m.lower() == month.lower()), month)

        return year, month_capitalized

    # Match "YYYY"
    year = section.strip()
    if year.isdigit():
        return year, "Janvier"

    raise ValueError(f"Could not extract year and month from section: {section}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    archive = get_year_links_from_archive(MAIN_URL)

    all_facts = []
    for year, data in sorted(archive.items()):
        print(f"=== {year} ===")
        print(f" Parsing {year}:", end='')
        if data['exists']:
            facts = parse_year_facts(data['url'], year)
            print(f" {len(facts)} facts")
            all_facts.extend(facts)
        else:
            print(" no page")

    grouped_data = defaultdict(lambda: defaultdict(list))
    for fact in all_facts:
        year, month = _extract_year_and_month_from_section(fact['section'])
        grouped_data[year][month].append(fact)

    out_path = os.path.join(OUTPUT_DIR, 'all_facts.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        # Sort by year, then by month index to ensure chronological order.
        sorted_grouped_data = {
            y: {
                m: grouped_data[y][m]
                for m in FRENCH_MONTHS if m in grouped_data[y]
            }
            for y in sorted(grouped_data.keys())
        }
        json.dump(sorted_grouped_data, f, ensure_ascii=False, indent=2)

    total_facts = len(all_facts)
    print(f"\nDone! All data written to {out_path}, total facts collected: {total_facts}.")


if __name__ == "__main__":
    main()
