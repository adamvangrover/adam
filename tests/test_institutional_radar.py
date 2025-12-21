import unittest

from core.institutional_radar.analytics import InstitutionalRadarAnalytics
from core.institutional_radar.database import FundMasterDB, SecurityMasterDB, SessionLocal, init_db
from core.institutional_radar.ingestion import SECEdgarScraper


class TestInstitutionalRadar(unittest.TestCase):
    def setUp(self):
        # Initialize DB
        init_db()
        self.session = SessionLocal()

        # Populate Seed Data (Fund Master, Security Master)
        if not self.session.query(FundMasterDB).filter_by(cik='0001067983').first():
            f1 = FundMasterDB(cik='0001067983', fund_name='BERKSHIRE HATHAWAY INC', fund_style='Family Office')
            f2 = FundMasterDB(cik='0000000001', fund_name='MOCK FUND A', fund_style='Hedge Fund')
            self.session.add(f1)
            self.session.add(f2)

            s1 = SecurityMasterDB(cusip='AAPL', ticker='AAPL', name='Apple Inc', sector='Technology')
            s2 = SecurityMasterDB(cusip='MSFT', ticker='MSFT', name='Microsoft', sector='Technology')
            self.session.add(s1)
            self.session.add(s2)
            self.session.commit()

    def tearDown(self):
        self.session.close()

    def test_ingestion_mock(self):
        scraper = SECEdgarScraper(mock_mode=True)
        # Verify it runs without error
        try:
            scraper.run_pipeline(['0001067983'], 2025, 3)
        except Exception as e:
            self.fail(f"Ingestion pipeline failed: {e}")

    def test_analytics(self):
        analytics = InstitutionalRadarAnalytics(self.session)
        # Mocking data via analytics requires data in DB.
        # Since ingestion mock puts data, we can try to query it.
        # But ingestion mock in 'ingestion.py' might not insert meaningful data for analytics
        # if the XML parsing returns empty or dummy.
        # Let's manually insert a holding for testing analytics.

        # We need a filing first
        import uuid
        from datetime import date

        from core.institutional_radar.database import FilingEventDB, HoldingDetailDB

        filing_id = uuid.uuid4()
        fe = FilingEventDB(
            filing_id=filing_id,
            cik='0000000001',
            report_period=date(2025, 9, 30),
            filing_date=date(2025, 11, 14),
            accession_number='000-000',
            is_amendment=False
        )
        self.session.add(fe)

        hd = HoldingDetailDB(
            holding_id=uuid.uuid4(),
            filing_id=filing_id,
            cusip='AAPL',
            shares=1000,
            value=200000,
            put_call=None
        )
        self.session.add(hd)
        self.session.commit()

        # Run Analytics
        df = analytics.get_quarterly_holdings(2025, 3)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['ticker'], 'AAPL')

        crowding = analytics.calculate_crowding_score(2025, 3)
        self.assertFalse(crowding.empty)
        # Crowding should be calculated for AAPL
        # Breadth N=1 (Mock Fund A)

        # Test Flows (needs previous quarter)
        # Insert Q2 data
        fe2 = FilingEventDB(
            filing_id=uuid.uuid4(),
            cik='0000000001',
            report_period=date(2025, 6, 30),
            filing_date=date(2025, 8, 14),
            accession_number='000-001',
            is_amendment=False
        )
        self.session.add(fe2)
        hd2 = HoldingDetailDB(
            holding_id=uuid.uuid4(),
            filing_id=fe2.filing_id,
            cusip='AAPL',
            shares=500,
            value=100000,
            put_call=None
        )
        self.session.add(hd2)
        self.session.commit()

        flows = analytics.calculate_sector_flows(2025, 3)
        # Net flow = 200k - 100k = 100k
        self.assertFalse(flows.empty)
        self.assertEqual(flows.iloc[0]['net_flow'], 100000)

if __name__ == '__main__':
    unittest.main()
