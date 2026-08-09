import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import os
    import re
    import json
    import requests
    from datetime import datetime
    from typing import Any
    from collections import defaultdict
    from bs4 import BeautifulSoup, Tag
    from urllib.parse import urljoin, unquote
    BASE_URL = 'https://vi.wikipedia.org'
    MAIN_URL = BASE_URL + '/wiki/Wikipedia:B%E1%BA%A1n_c%C3%B3_bi%E1%BA%BFt'
    OUTPUT_DIR = 'data/vi'
    NUMBER_TO_MONTH = {1: 'Tháng một', 2: 'Tháng hai', 3: 'Tháng ba', 4: 'Tháng tư', 5: 'Tháng năm', 6: 'Tháng sáu', 7: 'Tháng bảy', 8: 'Tháng tám', 9: 'Tháng chín', 10: 'Tháng mười', 11: 'Tháng mười một', 12: 'Tháng mười hai'}

    def calculate_month_number_from_week_number(week_number: int) -> int:
        """
        Calculate the approximate month number from the week number.
        """
        return (week_number - 1) * 7 // 30 + 1

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
            raise ValueError('Could not find the div.')
        archive_links = {}
        for a_tag in div.find_all('a', href=True):
            _year = a_tag.text.strip()
            if re.match('^\\d{4}$', _year):
                href = a_tag['href']
                full_url = urljoin(BASE_URL, href)
                exists = 'new' not in a_tag.get('class', [])
                archive_links[_year] = {'url': full_url, 'exists': exists}
        return archive_links
    return (
        Any,
        BeautifulSoup,
        MAIN_URL,
        Tag,
        get_year_links_from_archive,
        re,
        requests,
        unquote,
        urljoin,
    )


@app.cell
def _(MAIN_URL, get_year_links_from_archive):
    archive = get_year_links_from_archive(MAIN_URL)

    archive
    return (archive,)


@app.cell
def _(Any, BeautifulSoup, Tag, requests, unquote, urljoin):
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
    return (parse_year_facts,)


@app.cell
def _(archive, parse_year_facts):
    _year = '2025'
    facts = parse_year_facts(archive[_year]['url'])
    facts
    return


@app.cell
def _(re):
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
    return (extract_month_from_title,)


@app.cell
def _(extract_month_from_title):
    extract_month_from_title('3 22 грудня 2022 — 23 січня 2023')
    return


@app.cell
def _():
    import dateparser
    _year = '2025'
    month = '7'
    section = '11 — 25 лютого 2025'
    dt_month = dateparser.parse(section)
    fact_date = dt_month.strftime('%Y-%m') if dt_month else None
    fact_date
    return


@app.cell
def _():
    a = '2015-07-27'
    a.split('-')
    return


@app.cell
def _():
    ''
    return


if __name__ == "__main__":
    app.run()

