import os
import re
import json
import requests
from datetime import datetime
from typing import Any
from collections import defaultdict

from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, unquote

BASE_URL = 'https://sv.wikipedia.org'
MAIN_URL = BASE_URL + '/wiki/Wikipedia:Visste_du_att'

OUTPUT_DIR = 'data/sv'

DEFAULT_SECTION_MONTH = 'Januari'


def get_year_links_from_archive(main_page_url: str) -> dict[str, dict[str, Any]]:
    """
    Get the year links from the main archive page.
    Current year link is the main page.
    """
    resp = requests.get(main_page_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    table = soup.find('td', class_='navbox-list navbox-odd')
    if not table:
        raise ValueError("Could not find the table.")

    archive_links = {}
    for a_tag in table.find_all('a', href=True):        
        year = a_tag.text.strip()
        if re.match(r'^\d{4}$', year):
            href = a_tag['href']
            full_url = urljoin(BASE_URL, href)
            exists = 'new' not in a_tag.get('class', [])
            archive_links[year] = {
                'url': full_url,
                'exists': exists
            }

    # Current year link is the main page.
    current_year = str(datetime.now().year)
    archive_links[current_year] = {
        'url': main_page_url,
        'exists': True,
    }

    return archive_links


def _extract_fact_data(element: Tag, base_url: str) -> dict[str, Any]:
    """Extracts text, links, and relevant links from a BeautifulSoup Tag."""
    fact_text = "Visste du att " + element.get_text(" ", strip=True).lstrip().lstrip('.').lstrip('…').lstrip()
    fact_text = fact_text.replace('(se bild)', '').replace('( se bild )', '').replace('  ', ' ')
    fact_text = fact_text.replace('\xa0', ' ')

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


def parse_year_facts(year_url: str, year: str) -> list[dict]:
    """Parse the facts from the year page."""
    resp = requests.get(year_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results: list[dict] = []

    # Parse facts from all uls in section div
    div = soup.find("div", class_="mw-content-ltr mw-parser-output")
    if not div:
        return []
    
    # Different parsing for different years.
    if int(year) <= 2010: 
        ps = div.find_all("p", recursive=False)

        for p in ps:
            html_content = p.decode_contents()
            
            # Split by <br> tags, handling multiple separators and surrounding whitespace.
            fact_htmls = re.split(r'(?:<br[^>]*>\s*)+', html_content)
            
            # Filter out empty strings that may result from leading/trailing <br>s
            fact_htmls = [html.strip() for html in fact_htmls if html.strip()]

            for fact_html in fact_htmls:
                # Create a temporary soup to parse the fact snippet
                fact_soup = BeautifulSoup(f"<span>{fact_html}</span>", "html.parser").span
                if not fact_soup:
                    continue
                
                fact_data = _extract_fact_data(fact_soup, year_url)
                section = DEFAULT_SECTION_MONTH + ' ' + year

                if fact_data['text']:
                    results.append({
                        "section": section,
                        **fact_data
                    })
    elif int(year) <= 2012:
        for elem in div:
            if not (isinstance(elem, Tag) and elem.name == "ul"):
                continue

            for li in elem.find_all("li"):
                fact_data = _extract_fact_data(li, year_url)
                section = DEFAULT_SECTION_MONTH + ' ' + year

                if fact_data['text']:
                    results.append({
                        "section": section,
                        **fact_data
                    })
    else:
        current_section_name = DEFAULT_SECTION_MONTH + ' ' + year
        for elem in div:
            if not (isinstance(elem, Tag) and (elem.name == 'div' or elem.name == 'ul')):
                continue
            
            # End of facts
            if (
                elem.name == 'div' and 
                'mw-heading' in elem.get('class', []) and 
                'mw-heading2' in elem.get('class', []) and 
                'ext-discussiontools-init-section' in elem.get('class', [])
            ):
                h2 = elem.find('h2')
                if h2 and h2.text.strip() == 'Se även':
                    break
        
            # New section
            if elem.name == 'div' and 'mw-heading3' in elem.get('class', []):
                h3 = elem.find('h3')
                if h3:
                    current_section_name = h3.text.strip()

            # Fact
            elif elem.name == 'ul':
                for li in elem.find_all('li'):
                    fact_data = _extract_fact_data(li, year_url)

                    if fact_data['text']:
                        results.append({
                            "section": current_section_name,
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
                month_name = fact['section'].split()[0]
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
