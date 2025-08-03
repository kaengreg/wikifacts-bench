import os
import json
import requests
from typing import Any

from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, unquote

BASE_URL = 'https://pl.wikipedia.org'
MAIN_URL = BASE_URL + '/wiki/Wikiprojekt:Czy_wiesz/archiwum'
OUTPUT_DIR = 'data/pl'

MONTH_NUMBERS_TO_NAMES = {
    '01': 'styczeń',
    '02': 'luty',
    '03': 'marzec',
    '04': 'kwiecień',
    '05': 'maj',
    '06': 'czerwiec',
    '07': 'lipiec',
    '08': 'sierpień',
    '09': 'wrzesień',
    '10': 'październik',
    '11': 'listopad',
    '12': 'grudzień',
}


def get_month_links_from_archive(main_page_url):
    resp = requests.get(main_page_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    # Find tables with month links
    t1 = soup.find('table', class_='a1 inner-standard')
    if not t1:
        raise ValueError("Could not find the table with class 'a1 inner-standard'.")
    t2 = soup.find('table', class_='a2 inner-standard')
    if not t2:
        raise ValueError("Could not find the table with class 'a2 inner-standard'.")
    t3 = soup.find('table', class_='a3 inner-standard')
    if not t3:
        raise ValueError("Could not find the table with class 'a3 inner-standard'.")
    t4 = soup.find('table', class_='a4 inner-standard')
    if not t4:
        raise ValueError("Could not find the table with class 'a4 inner-standard'.")

    archive = {}

    # Years 2004-2007
    for tr in t1.find_all('tr'):
        header = tr.find('th').get_text(strip=True)
        date_parts = header.split(':')[0].split('-')
        year, month = date_parts[0], date_parts[1]
        month_name = MONTH_NUMBERS_TO_NAMES[month]

        a = tr.find('a')
        url = a['href']
        full_url = urljoin(BASE_URL, url)
        exists = 'new' not in a.get('class', [])

        if year not in archive:
            archive[year] = []

        archive[year].append({
            'month': month_name,
            'url': full_url,
            'exists': exists
        })

    # Years 2008-2025 (2025 is not complete)
    for ti in [t2, t3, t4]:
        for tr in ti.find_all('tr'):
            year = tr.find('th').get_text(strip=True)

            month_list = tr.find('ul')
            for li in month_list.find_all('li'):
                a = li.find('a')
                month_name = a.get_text(strip=True)
                url = a['href']
                full_url = urljoin(BASE_URL, url)
                exists = 'new' not in a.get('class', [])

                if year not in archive:
                    archive[year] = []

                archive[year].append({
                    'month': month_name,
                    'url': full_url,
                    'exists': exists
                })
     
    return archive        


def _extract_fact_data(element: Tag, base_url: str) -> dict[str, Any]:
    """Extracts text, links, and relevant links from a BeautifulSoup Tag."""
    raw_text = element.get_text(" ", strip=True).replace('Czy wiesz ...', '').lstrip().lstrip('.').lstrip('…').lstrip()
    if not raw_text:
        return {}
    if (
        raw_text.startswith('Strona ekspozycji') or
        raw_text.startswith('Z nowych artykułów w Wikipedii') or
        raw_text.startswith('Z nowych i ostatnio rozbudowanych artykułów')
    ):
        return {}
    
    fact_text = "Czy wiesz " + raw_text
    fact_text = fact_text.replace('  ', ' ')
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


def parse_month_facts(month_url: str, month_name: str, year: str) -> list[dict]:
    resp = requests.get(month_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results: list[dict] = []

    # Different parsing for different years
    if int(year) < 2007 or (int(year) == 2007 and month_name != 'maj'):
        main_div = soup.find("div", class_="mw-heading mw-heading2 ext-discussiontools-init-section")
        div = main_div.find_next_sibling("div")

        for p in div.find_all("p"):
            fact_data = _extract_fact_data(p, month_url)

            if fact_data and fact_data['text']:
                results.append({
                    "section": month_name + ' ' + year,
                    **fact_data
                })
    elif int(year) == 2007 and month_name == 'maj':
        for main_div in soup.find_all("div", class_="mw-heading mw-heading2 ext-discussiontools-init-section"):
            section = main_div.find('h2').get_text(strip=True)

            for sib in main_div.find_next_siblings():
                if sib.name == "div" and "mw-heading" in sib.get("class", []):
                    break
                
                if sib.name == 'p':
                    fact_data = _extract_fact_data(sib, month_url)

                    if fact_data and fact_data['text']:
                        results.append({
                            "section": section,
                            **fact_data
                        })
    # For years > 2007
    else:
        for main_div in soup.find_all("div", class_="mw-heading mw-heading3"):
            section = main_div.find('h3').get_text(strip=True)

            for sib in main_div.find_next_siblings():
                if sib.name == "div" and "mw-heading" in sib.get("class", []):
                    break
                
                if sib.name == 'p':
                    fact_data = _extract_fact_data(sib, month_url)
                    
                    if fact_data and fact_data['text']:
                        results.append({
                            "section": section,
                            **fact_data
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
                facts = parse_month_facts(m['url'], month_name, year)
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
