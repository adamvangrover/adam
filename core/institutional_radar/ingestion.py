import requests
import pandas as pd
import time
import logging
import uuid
from io import BytesIO
from bs4 import BeautifulSoup
from typing import List, Optional, Dict
from core.utils.logging_utils import get_logger
from core.institutional_radar.schema import HoldingDetail, FilingEvent
from core.institutional_radar.database import SessionLocal, FilingEventDB, HoldingDetailDB, FundMasterDB, init_db
from datetime import datetime, date

logger = get_logger("institutional_radar.ingestion")

class SECEdgarScraper:
    BASE_URL = "https://www.sec.gov/Archives"
    HEADERS = {"User-Agent": "Adam-Bot/1.0 (internal@adam.finance)"}

    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode

    def _get(self, url: str) -> requests.Response:
        if self.mock_mode:
            logger.info(f"[MOCK] Fetching {url}")
            resp = requests.Response()
            resp.status_code = 200
            resp._content = b""
            return resp

        time.sleep(0.1) # Rate limit compliance (10 req/sec max)
        logger.info(f"Fetching {url}")
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=10)
            resp.raise_for_status()
            return resp
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            raise

    def fetch_master_index(self, year: int, quarter: int) -> pd.DataFrame:
        """
        Downloads and parses the master.idx file for a given quarter.
        """
        url = f"{self.BASE_URL}/edgar/full-index/{year}/QTR{quarter}/master.idx"
        if self.mock_mode:
            # Return a dummy dataframe
            return pd.DataFrame({
                'CIK': ['0001067983', '0001067983'],
                'Company Name': ['BERKSHIRE HATHAWAY INC', 'BERKSHIRE HATHAWAY INC'],
                'Form Type': ['13F-HR', '10-K'],
                'Date Filed': ['2025-11-14', '2025-02-14'],
                'Filename': ['edgar/data/1067983/0001067983-25-000001.txt', 'edgar/data/1067983/0001067983-25-000002.txt']
            })

        response = self._get(url)
        # Skip the header lines (usually top 10 lines)
        content = response.content.decode('latin-1')
        lines = content.splitlines()

        # Find start of data
        start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("CIK"):
                start_idx = i + 2
                break

        data = []
        for line in lines[start_idx:]:
            parts = line.split('|')
            if len(parts) == 5:
                data.append(parts)

        df = pd.DataFrame(data, columns=['CIK', 'Company Name', 'Form Type', 'Date Filed', 'Filename'])
        return df

    def get_filing_xml_url(self, archive_url: str) -> Optional[str]:
        """
        Given a text archive URL (e.g., .../0000.txt), find the infotable.xml link.
        Usually we need to go to the index page .../0000-index.html
        """
        # archive_url is like edgar/data/CIK/ACCESSION.txt
        # index_url is like edgar/data/CIK/ACCESSION-index.html
        # But actually, the Filename in master.idx is the full path to the text dump.
        # We want the folder listing.

        if self.mock_mode:
            return "http://mock.url/infotable.xml"

        base_path = archive_url.replace('.txt', '-index.html')
        # Wait, the master.idx Filename is usually edgar/data/{cik}/{accession}.txt
        # The index page is typically constructed by taking the accession number (without dashes) as a folder?
        # Actually, standard practice is to visit the -index.html page.

        index_url = f"{self.BASE_URL}/{base_path}"
        resp = self._get(index_url)
        soup = BeautifulSoup(resp.content, 'html.parser')

        # Look for the XML file in the table
        # We look for "xml" in the document name and usually type "INFORMATION TABLE"
        for row in soup.find_all('tr'):
            cols = row.find_all('td')
            if len(cols) > 3:
                doc_type = cols[1].text.strip()
                doc_href = cols[2].find('a')['href']
                if 'INFORMATION TABLE' in doc_type and doc_href.endswith('.xml'):
                    return f"{self.BASE_URL}{doc_href}"
                # Fallback: sometimes it's labeled 13F-HR
                if '13F-HR' in doc_type and doc_href.endswith('.xml'):
                    return f"{self.BASE_URL}{doc_href}"

        return None

    def parse_infotable_xml(self, xml_content: bytes) -> List[HoldingDetail]:
        """
        Parses the XML content of a 13F Information Table.
        """
        try:
            soup = BeautifulSoup(xml_content, 'xml')
        except:
            soup = BeautifulSoup(xml_content, 'html.parser')

        holdings = []

        # Determine if we need to parse namespace
        # Standard tags: nameOfIssuer, cusip, value, shrsOrPrnAmt -> sshPrnamt

        info_tables = soup.find_all(['infoTable', 'ns1:infoTable'])

        for table in info_tables:
            try:
                # Helper to find tag text with or without namespace
                def get_text(tag_name):
                    t = table.find(tag_name)
                    if not t:
                        t = table.find(f"ns1:{tag_name}")
                    return t.text if t else None

                def get_nested_text(parent, tag_name):
                    p = table.find(parent)
                    if not p:
                        p = table.find(f"ns1:{parent}")
                    if p:
                        t = p.find(tag_name)
                        if not t:
                            t = p.find(f"ns1:{tag_name}")
                        return t.text if t else None
                    return None

                name = get_text('nameOfIssuer')
                cusip = get_text('cusip')
                val_text = get_text('value')
                value = int(val_text) if val_text else 0

                # Shares
                shares_tag = table.find('shrsOrPrnAmt') or table.find('ns1:shrsOrPrnAmt')
                shares = 0
                if shares_tag:
                    ssh_tag = shares_tag.find('sshPrnamt') or shares_tag.find('ns1:sshPrnamt')
                    if ssh_tag and ssh_tag.text:
                        shares = int(ssh_tag.text)

                put_call = get_text('putCall')

                # Voting
                vote_tag = table.find('votingAuthority') or table.find('ns1:votingAuthority')
                vote_sole = 0
                if vote_tag:
                    sole_tag = vote_tag.find('Sole') or vote_tag.find('ns1:Sole') # Case sensitive? usually Sole
                    if not sole_tag:
                         sole_tag = vote_tag.find('sole')
                    if sole_tag and sole_tag.text:
                        vote_sole = int(sole_tag.text)

                if cusip and shares > 0:
                     # Create HoldingDetail object (Pydantic)
                     h = HoldingDetail(
                         holding_id=uuid.uuid4(),
                         filing_id=uuid.uuid4(), # Temporary placeholder
                         cusip=cusip.upper(),
                         ticker=None, # Needs mapping
                         shares=shares,
                         value=value,
                         put_call=put_call if put_call in ['PUT', 'CALL'] else None,
                         vote_sole=vote_sole
                     )
                     holdings.append(h)
            except Exception as e:
                logger.warning(f"Failed to parse a row in 13F XML: {e}")
                continue

        return holdings

    def run_pipeline(self, ciks: List[str], year: int, quarter: int):
        """
        Orchestrates the ingestion for a list of CIKs.
        """
        init_db()
        session = SessionLocal()

        logger.info(f"Starting pipeline for Y{year} Q{quarter}")
        try:
            df = self.fetch_master_index(year, quarter)

            # Filter for 13F-HR and CIKs
            target_filings = df[
                (df['Form Type'] == '13F-HR') &
                (df['CIK'].isin(ciks))
            ]

            logger.info(f"Found {len(target_filings)} filings to process.")

            for _, row in target_filings.iterrows():
                cik = row['CIK']
                filename = row['Filename']
                date_filed = row['Date Filed']

                # Check if already exists
                existing = session.query(FilingEventDB).filter_by(
                    cik=cik,
                    filing_date=datetime.strptime(date_filed, '%Y-%m-%d').date()
                ).first()

                if existing:
                    logger.info(f"Skipping existing filing for CIK {cik} on {date_filed}")
                    continue

                # Get XML URL
                xml_url = self.get_filing_xml_url(filename)
                if not xml_url:
                    logger.warning(f"No XML found for {filename}")
                    continue

                # Download and Parse
                if self.mock_mode:
                    xml_content = b"<xml>...</xml>" # Dummy
                    parsed_holdings = []
                else:
                    xml_resp = self._get(xml_url)
                    parsed_holdings = self.parse_infotable_xml(xml_resp.content)

                # Save to DB
                filing_db = FilingEventDB(
                    cik=cik,
                    report_period=date(year, quarter*3, 30), # Approximation
                    filing_date=datetime.strptime(date_filed, '%Y-%m-%d').date(),
                    accession_number=filename.split('/')[-1].replace('.txt', ''),
                    is_amendment=False
                )
                session.add(filing_db)
                session.flush() # Generate ID

                for h in parsed_holdings:
                    h_db = HoldingDetailDB(
                        filing_id=filing_db.filing_id,
                        cusip=h.cusip,
                        ticker=h.ticker,
                        shares=h.shares,
                        value=h.value,
                        put_call=h.put_call,
                        vote_sole=h.vote_sole
                    )
                    session.add(h_db)

                session.commit()
                logger.info(f"Successfully ingested filing for {cik}")

        except Exception as e:
            session.rollback()
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            session.close()
