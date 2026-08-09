import os
import json
import requests
import re

from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote

BASE_URL = 'https://uk.wikipedia.org'
MAIN_URL = BASE_URL + '/wiki/%D0%92%D1%96%D0%BA%D1%96%D0%BF%D0%B5%D0%B4%D1%96%D1%8F:%D0%9F%D1%80%D0%BE%D1%94%D0%BA%D1%82:%D0%A7%D0%B8_%D0%B2%D0%B8_%D0%B7%D0%BD%D0%B0%D1%94%D1%82%D0%B5/%D0%90%D1%80%D1%85%D1%96%D0%B2_%D1%80%D1%83%D0%B1%D1%80%D0%B8%D0%BA%D0%B8'
OUTPUT_DIR = 'data/uk'

MONTHS = {
    'січень': 'січень', 'січня': 'січень',
    'лютий': 'лютий', 'лютого': 'лютий',
    'березень': 'березень', 'березня': 'березень',
    'квітень': 'квітень', 'квітня': 'квітень',
    'травень': 'травень', 'травня': 'травень',
    'червень': 'червень', 'червня': 'червень',
    'липень': 'липень', 'липня': 'липень',
    'серпень': 'серпень', 'серпня': 'серпень',
    'вересень': 'вересень', 'вересня': 'вересень',
    'жовтень': 'жовтень', 'жовтня': 'жовтень',
    'листопад': 'листопад', 'листопада': 'листопад',
    'грудень': 'грудень', 'грудня': 'грудень',
}


def extract_month_from_title(title: str) -> str | None:
    """Extracts the first mentioned month from a section title."""
    words = re.split(r'[\s—-]+', title)

    for word in words:
        cleaned_word = word.lower().strip()
        
        if cleaned_word in MONTHS:
            return MONTHS[cleaned_word]

    return None


def extract_year_from_title(title: str) -> int | None:
    """Extracts the first mentioned year (4-digit number) from a section title."""
    match = re.search(r'\b(\d{4})\b', title)
    if match:
        return int(match.group(1))
    return None


def get_year_quarter_links_from_archive(main_page_url):
    resp = requests.get(main_page_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    div = soup.find('div', class_='mw-heading mw-heading2 ext-discussiontools-init-section')
    archive_ul = div.find_next('ul')

    archive = {}
    for li in archive_ul.find_all('li', recursive=False):
        year_b = li.find('b')
        year = year_b.text.strip().split()[0]

        quarters = []
        for a in li.find_all('a', recursive=False):
            quarter_name = a.text.strip()
            href = a.get('href')
            full_url = BASE_URL + href
            exists = 'new' not in a.get('class', [])

            quarters.append({
                'quarter': quarter_name,
                'url':   full_url,
                'exists': exists
            })
        
        archive[year] = quarters

    return archive


def parse_quarter_facts(quarter_url: str, default_section_name: str) -> list[dict]:
    resp = requests.get(quarter_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results: list[dict] = []

    # Find all section divs that act as headers for facts
    section_divs = soup.find_all("div", class_="mw-heading mw-heading2 ext-discussiontools-init-section")
    
    is_simple_case = len(section_divs) == 1

    for div in section_divs:
        header = div.find('h2')
        section_name_text = header.get_text(strip=True) if header else ""

        # The rule is to skip the div with this specific title, and all its contents.
        if section_name_text == "Архів по місяцях" and not is_simple_case:
            continue

        section_name = default_section_name
        # We extract titles unless it's the simple, single-div case.
        if not is_simple_case and section_name_text and section_name_text != "Архів по місяцях":
            section_name = section_name_text

        uls_for_this_section = []
        # Collect all 'ul' siblings that appear after the current 'div' and before the next one
        for sibling in div.find_next_siblings():
            if sibling.name == 'div' and "mw-heading" in sibling.get('class', []):
                break
            if sibling.name == 'ul':
                uls_for_this_section.append(sibling)

        # In the original simple case (only one section div on the page), skip the first 'ul'
        if is_simple_case and uls_for_this_section:
            uls_to_process = uls_for_this_section[1:]
        else:
            uls_to_process = uls_for_this_section

        for ul in uls_to_process:
            for li in ul.find_all('li', recursive=False):
                fact_text = li.get_text(" ", strip=True).replace('\xa0', ' ').strip()

                links: list[str] = []
                relevant_links: list[str] = []

                for a in li.find_all("a", href=True):
                    href = a["href"]
                    if not href.startswith("/wiki/"):
                        continue

                    full_url = unquote(urljoin(quarter_url, href))

                    if a.find_parent('b') or a.find('b'):
                        relevant_links.append(full_url)

                    links.append(full_url)

                results.append({
                    "section": section_name,
                    "text": fact_text,
                    "links": links,
                    "relevant_links": relevant_links,
                })

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    archive = get_year_quarter_links_from_archive(MAIN_URL)

    all_data: dict[str, dict[str, list[dict]] ] = {}

    for archive_year, quarters in archive.items():
        print(f"=== {archive_year} ===")

        for q in quarters:
            quarter_name = q['quarter']
            print(f" Парсим {archive_year} — {quarter_name}:", end=' ')

            if q['exists']:
                default_section_name = f"{quarter_name} {archive_year}"
                facts = parse_quarter_facts(q['url'], default_section_name)
                print(f"{len(facts)} фактов")

                for fact in facts:
                    # Determine the correct year, preferring the one in the title
                    year_from_title = extract_year_from_title(fact['section'])
                    final_year_str = str(year_from_title or archive_year)
                    
                    # Determine the month or fallback to the quarter name
                    month_name = extract_month_from_title(fact['section'])
                    group_key = month_name if month_name else quarter_name

                    # Populate all_data with the correct year and group
                    if final_year_str not in all_data:
                        all_data[final_year_str] = {}
                    
                    if group_key not in all_data[final_year_str]:
                        all_data[final_year_str][group_key] = []
                    
                    all_data[final_year_str][group_key].append(fact)

            else:
                facts = []
                print("нет страницы")

    out_path = os.path.join(OUTPUT_DIR, 'all_facts.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\nГотово! Все данные записаны в {out_path}, всего собрано {sum(len(facts) for quarters in all_data.values() for facts in quarters.values())} фактов.")


if __name__ == "__main__":
    main()
