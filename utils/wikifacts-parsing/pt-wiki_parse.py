import os
import re
import json
import requests
from typing import Any
from collections import defaultdict

from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, unquote

BASE_URL = 'https://pt.wikipedia.org'
MAIN_URL = BASE_URL + '/wiki/Wikip%C3%A9dia:Sabia_que'

OUTPUT_DIR = 'data/pt'

PORTUGUESE_MONTHS = [
    'Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
    'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'
]


def get_year_links_from_archive(main_page_url: str) -> dict[str, dict[str, Any]]:
    """Get the year links from the main archive page."""
    resp = requests.get(main_page_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    table = soup.find('table', class_='tmbox tmbox-notice }}')
    if not table:
        raise ValueError("Could not find the table in the container div.")

    archive_links = {}
    for b_tag in table.find_all('b'):
        a_tag = b_tag.find('a', href=True)
        if a_tag:
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
    fact_text = element.get_text(" ", strip=True)

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


def _post_process_section_title(title: str, year: str) -> str:
    """Post-processes the section title to include the year."""
    if title == str(None):
        return year
    
    if len(title) < 4 or not title[-4:].isdigit():
        return f"{title} de {year}"
    
    return title


def _extract_month_from_section(section_title: str) -> str:
    """
    Extracts the month name from the section title, searching anywhere in the string.
    Defaults to 'Janeiro' if no month is found.
    """
    for month in PORTUGUESE_MONTHS:
        if re.search(r'\b' + re.escape(month) + r'\b', section_title, re.IGNORECASE):
            return month
    return "Janeiro"


def parse_year_facts(year_url: str, year: str) -> list[dict]:
    """Parse the facts from the year page."""
    resp = requests.get(year_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results: list[dict] = []

    # If a wikitable exists, it's the only element with facts.
    wikitable = soup.find("table", class_="wikitable")
    if wikitable:
        section_title = _post_process_section_title(str(None), year)
        for tr in wikitable.find_all("tr")[1:]:  # Ignore header row
            td = tr.find("td")
            if td:
                fact_data = _extract_fact_data(td, year_url)
                if fact_data['text']:
                    results.append({
                        "section": section_title,
                        **fact_data
                    })
        return results

    # Otherwise, parse facts from sections.
    section_divs = soup.select("div.mw-heading.mw-heading2, div.mw-heading.mw-heading3, div.mw-heading.mw-heading4")
    
    for section_div in section_divs:
        header = section_div.find(["h2", "h3", "h4"])
        raw_section_title = header.get_text(strip=True) if header else str(None)
        section_title = _post_process_section_title(raw_section_title, year)

        for sib in section_div.find_next_siblings():
            sib_classes = sib.get("class", []) if isinstance(sib, Tag) else []
            is_heading = "mw-heading" in sib_classes and any(f"mw-heading{i}" in sib_classes for i in [2, 3, 4])
            if is_heading:
                break

            # Some facts can be written as paragraphs
            if isinstance(sib, Tag) and sib.name == "p":
                fact_data = _extract_fact_data(sib, year_url)
                if fact_data['text']:
                    results.append({
                        "section": section_title,
                        **fact_data
                    })
                continue

            if not (isinstance(sib, Tag) and sib.name == "ul"):
                continue

            # Most facts are written as items in unordered lists
            for li in sib.find_all("li"):
                fact_data = _extract_fact_data(li, year_url)
                if fact_data['text']:
                    results.append({
                        "section": section_title,
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
            facts = parse_year_facts(data['url'], year)
            print(f" {len(facts)} facts")

            year_data: dict[str, list[dict]] = defaultdict(list)
            for fact in facts:
                month_name = _extract_month_from_section(fact['section'])
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
