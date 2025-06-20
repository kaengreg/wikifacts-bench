import os
import re
import json
import requests
import unicodedata
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, unquote

BASE_URL = 'https://ru.wikipedia.org'
MAIN_URL = BASE_URL + '/wiki/Проект:Знаете_ли_вы/Архив_рубрики'

OUTPUT_DIR = 'data/rus'

def preprocess_text(text: str) -> str:
    exclude_chars = "йё"
    text = re.sub(r'а́', 'а', text)
    text = re.sub(r"==\s*(.*?)\s*==\s*", r"\1 ", text)
    text = text.replace('\xa0', ' ').strip()
    text = unicodedata.normalize('NFC', text)
    result = []
    for char in text:
        if char in exclude_chars:
            result.append(char)
        else:
            base = unicodedata.normalize('NFD', char)
            no_diacritics = ''.join(c for c in base if not unicodedata.combining(c))
            result.append(no_diacritics)
    return ''.join(result)


def get_month_links_from_archive(main_page_url):
    resp = requests.get(main_page_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    content = soup.find('div', class_='ts-Box-description')

    archive_ul = None
    for ul in content.find_all('ul'):
        lis = ul.find_all('li', recursive=False)

        good = True
        for li in lis:
            bold = li.find('b')
            if bold is None or not re.match(r'^\d{4}\sгод[:\s]', bold.text.strip()):
                good = False
                break

        if good:
            archive_ul = ul
            break

    archive = {}
    for li in archive_ul.find_all('li', recursive=False):
        year_b = li.find('b')
        year = year_b.text.strip().split()[0]  
        months = []
        for a in li.find_all('a', recursive=False):
            month_name = a.text.strip()
            href       = a.get('href')
            full_url   = BASE_URL + href
            exists     = 'new' not in a.get('class', [])
            months.append({
                'month': month_name,
                'url':   full_url,
                'exists': exists
            })
        archive[year] = months

    return archive


def parse_month_facts(month_url: str) -> list[dict]:
    resp = requests.get(month_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results: list[dict] = []

    section_divs = soup.find_all("div", class_="ext-discussiontools-init-section")

    for section_div in section_divs:
        header = section_div.find("h2")
        section_title = header.get_text(strip=True) if header else "(без заголовка)"

        for sib in section_div.next_siblings:
            if isinstance(sib, Tag) and "ext-discussiontools-init-section" in sib.get("class", []):
                break

            if not isinstance(sib, Tag):
                continue

            for li in sib.find_all("li"):
                fact_text = li.get_text(" ", strip=True)

                links: list[str] = []
                relevant_links: list[str] = []

                for a in li.find_all("a", href=True):
                    href = a["href"]
                    if not href.startswith("/wiki/"):
                        continue
                
                    full_url = unquote(urljoin(month_url, href))

                    if a.find_parent('b'):
                        relevant_links.append(full_url)
                    else:
                        links.append(full_url)

                results.append({
                    "section": section_title,
                    "text": preprocess_text(fact_text),
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



