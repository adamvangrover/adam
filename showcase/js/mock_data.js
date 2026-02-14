window.MOCK_DATA = {
  "stats": {
    "version": "23.5",
    "status": "HYBRID_ONLINE",
    "cpu_load": 12,
    "memory_usage": 34,
    "active_tasks": 4
  },
  "files": [],
  "financial_data": {
    "synthetic_stock_data.csv": [
      {
        "time": "0.1",
        "value": 0.5
      },
      {
        "time": "0.2",
        "value": 0.51
      },
      {
        "time": "0.3",
        "value": 0.52
      },
      {
        "time": "0.4",
        "value": 0.53
      },
      {
        "time": "0.5",
        "value": 0.54
      },
      {
        "time": "0.6",
        "value": 0.55
      }
    ],
    "synthetic_black_swan_scenario.csv": []
  },
  "credit_library": [
    {
      "id": "Apple_Inc",
      "borrower_name": "Apple Inc.",
      "ticker": "AAPL",
      "sector": "Technology",
      "report_date": "2026-02-14T01:37:17.030616",
      "risk_score": 75.0,
      "file": "credit_memo_Apple_Inc.json",
      "summary": "The borrower Apple Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($148000.0M) but elevated leverage (0.6x).  [Ref: A..."
    },
    {
      "id": "Microsoft_Corp",
      "borrower_name": "Microsoft Corp",
      "ticker": "MSFT",
      "sector": "Technology",
      "report_date": "2026-02-14T01:37:17.035685",
      "risk_score": 75.0,
      "file": "credit_memo_Microsoft_Corp.json",
      "summary": "The borrower Microsoft Corp presents a mixed credit profile. Financial performance shows strong EBITDA ($165000.0M) but elevated leverage (0.2x).  [Re..."
    },
    {
      "id": "NVIDIA_Corp",
      "borrower_name": "NVIDIA Corp",
      "ticker": "NVDA",
      "sector": "Technology",
      "report_date": "2026-02-14T01:37:17.037134",
      "risk_score": 75.0,
      "file": "credit_memo_NVIDIA_Corp.json",
      "summary": "The borrower NVIDIA Corp presents a mixed credit profile. Financial performance shows strong EBITDA ($92000.0M) but elevated leverage (0.1x).  [Ref: T..."
    },
    {
      "id": "Alphabet_Inc",
      "borrower_name": "Alphabet Inc.",
      "ticker": "GOOGL",
      "sector": "Technology",
      "report_date": "2026-02-14T01:37:17.038187",
      "risk_score": 75.0,
      "file": "credit_memo_Alphabet_Inc.json",
      "summary": "The borrower Alphabet Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($138000.0M) but elevated leverage (0.1x).  [Ref..."
    },
    {
      "id": "Amazoncom_Inc",
      "borrower_name": "Amazon.com Inc.",
      "ticker": "AMZN",
      "sector": "Consumer",
      "report_date": "2026-02-14T01:37:17.039190",
      "risk_score": 75.0,
      "file": "credit_memo_Amazoncom_Inc.json",
      "summary": "The borrower Amazon.com Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($150000.0M) but elevated leverage (0.3x).  [R..."
    },
    {
      "id": "Tesla_Inc",
      "borrower_name": "Tesla Inc.",
      "ticker": "TSLA",
      "sector": "Consumer",
      "report_date": "2026-02-14T01:37:17.040466",
      "risk_score": 75.0,
      "file": "credit_memo_Tesla_Inc.json",
      "summary": "The borrower Tesla Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($32000.0M) but elevated leverage (0.2x).  [Ref: TS..."
    },
    {
      "id": "Meta_Platforms",
      "borrower_name": "Meta Platforms",
      "ticker": "META",
      "sector": "Technology",
      "report_date": "2026-02-14T01:37:17.041874",
      "risk_score": 75.0,
      "file": "credit_memo_Meta_Platforms.json",
      "summary": "The borrower Meta Platforms presents a mixed credit profile. Financial performance shows strong EBITDA ($110000.0M) but elevated leverage (0.3x).  [Re..."
    },
    {
      "id": "JPMorgan_Chase",
      "borrower_name": "JPMorgan Chase",
      "ticker": "JPM",
      "sector": "Financial",
      "report_date": "2026-02-14T01:37:17.043144",
      "risk_score": 45.0,
      "file": "credit_memo_JPMorgan_Chase.json",
      "summary": "The borrower JPMorgan Chase presents a mixed credit profile. Financial performance shows strong EBITDA ($105000.0M) but elevated leverage (10.2x).  [R..."
    },
    {
      "id": "Goldman_Sachs",
      "borrower_name": "Goldman Sachs",
      "ticker": "GS",
      "sector": "Financial",
      "report_date": "2026-02-14T01:37:17.044032",
      "risk_score": 45.0,
      "file": "credit_memo_Goldman_Sachs.json",
      "summary": "The borrower Goldman Sachs presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: Ge..."
    },
    {
      "id": "Bank_of_America",
      "borrower_name": "Bank of America",
      "ticker": "BAC",
      "sector": "Financial",
      "report_date": "2026-02-14T01:37:17.044911",
      "risk_score": 45.0,
      "file": "credit_memo_Bank_of_America.json",
      "summary": "The borrower Bank of America presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: ..."
    }
  ],
  "credit_memos": {
    "Meta_Platforms": {
      "borrower_name": "Meta Platforms",
      "report_date": "2026-02-14T01:37:17.041874",
      "executive_summary": "The borrower Meta Platforms presents a mixed credit profile. Financial performance shows strong EBITDA ($110000.0M) but elevated leverage (0.3x).  [Ref: TechCorp_10K_2025.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower Meta Platforms presents a mixed credit profile. Financial performance shows strong EBITDA ($110000.0M) but elevated leverage (0.3x).  [Ref: TechCorp_10K_2025.pdf]",
          "citations": [
            {
              "doc_id": "TechCorp_10K_2025.pdf",
              "chunk_id": "ffde1c1e-7ed3-4382-9f6b-1df1dbd29a3e",
              "page_number": 3
            }
          ],
          "author_agent": "Writer"
        },
        {
          "title": "Key Risks & Mitigants",
          "content": "Key Risks:\n",
          "citations": [],
          "author_agent": "Risk Officer"
        },
        {
          "title": "Financial Analysis",
          "content": "EBITDA: $110000.0M | Leverage: 0.3x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "financial_ratios": {
        "leverage_ratio": 0.2727272727272727,
        "dscr": 366.6666666666667,
        "current_ratio": 1.5,
        "revenue": 210000.0,
        "ebitda": 110000.0,
        "net_income": 66000.0
      },
      "historical_financials": [
        {
          "total_assets": 350000.0,
          "total_liabilities": 105000.0,
          "total_equity": 245000.0,
          "revenue": 210000.0,
          "ebitda": 110000.0,
          "net_income": 66000.0,
          "interest_expense": 300.0,
          "dscr": 366.6666666666667,
          "leverage_ratio": 0.2727272727272727,
          "current_ratio": 1.5,
          "period": "FY2026"
        },
        {
          "total_assets": 322000.0,
          "total_liabilities": 96600.0,
          "total_equity": 225400.0,
          "revenue": 189000.0,
          "ebitda": 96800.0,
          "net_income": 56100.0,
          "interest_expense": 300.0,
          "dscr": 366.6666666666667,
          "leverage_ratio": 0.2727272727272727,
          "current_ratio": 1.5,
          "period": "FY2025"
        },
        {
          "total_assets": 305900.0,
          "total_liabilities": 91770.0,
          "total_equity": 214130.0,
          "revenue": 173880.0,
          "ebitda": 87120.0,
          "net_income": 49368.0,
          "interest_expense": 300.0,
          "dscr": 366.6666666666667,
          "leverage_ratio": 0.2727272727272727,
          "current_ratio": 1.5,
          "period": "FY2024"
        }
      ],
      "dcf_analysis": {
        "free_cash_flow": [
          75075.0,
          78079.78749999999,
          80427.77600000001,
          82047.89454468747,
          82888.09631245
        ],
        "growth_rate": 0.03,
        "wacc": 0.09,
        "terminal_value": 1422912.3200303917,
        "enterprise_value": 1233491.090864101,
        "equity_value": 1128491.090864101,
        "share_price": 185.0
      },
      "risk_score": 75.0,
      "credit_ratings": [
        {
          "agency": "Moody's",
          "rating": "Ba2",
          "outlook": "Stable",
          "date": "2026-02-14"
        },
        {
          "agency": "S&P",
          "rating": "BB",
          "outlook": "Stable",
          "date": "2026-02-14"
        }
      ],
      "debt_facilities": [
        {
          "facility_type": "Revolver",
          "amount_committed": 100.0,
          "amount_drawn": 20.0,
          "interest_rate": "Prime + 1.0%",
          "maturity_date": "2026-01-01",
          "snc_rating": "Pass",
          "drc": 0.8,
          "ltv": 0.5,
          "conviction_score": 0.7,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 2.2,
              "principal": 2.0,
              "interest": 0.2,
              "remaining_balance": 18.0
            },
            {
              "year": 2028,
              "payment_amount": 2.18,
              "principal": 2.0,
              "interest": 0.18,
              "remaining_balance": 16.0
            },
            {
              "year": 2029,
              "payment_amount": 2.16,
              "principal": 2.0,
              "interest": 0.16,
              "remaining_balance": 14.0
            },
            {
              "year": 2030,
              "payment_amount": 2.14,
              "principal": 2.0,
              "interest": 0.14,
              "remaining_balance": 12.0
            },
            {
              "year": 2031,
              "payment_amount": 2.12,
              "principal": 2.0,
              "interest": 0.12,
              "remaining_balance": 10.0
            }
          ]
        },
        {
          "facility_type": "Term Loan",
          "amount_committed": 500.0,
          "amount_drawn": 500.0,
          "interest_rate": "5.50%",
          "maturity_date": "2029-01-01",
          "snc_rating": "Pass",
          "drc": 0.75,
          "ltv": 0.6,
          "conviction_score": 0.75,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 77.5,
              "principal": 50.0,
              "interest": 27.5,
              "remaining_balance": 450.0
            },
            {
              "year": 2028,
              "payment_amount": 74.75,
              "principal": 50.0,
              "interest": 24.75,
              "remaining_balance": 400.0
            },
            {
              "year": 2029,
              "payment_amount": 72.0,
              "principal": 50.0,
              "interest": 22.0,
              "remaining_balance": 350.0
            },
            {
              "year": 2030,
              "payment_amount": 69.25,
              "principal": 50.0,
              "interest": 19.25,
              "remaining_balance": 300.0
            },
            {
              "year": 2031,
              "payment_amount": 66.5,
              "principal": 50.0,
              "interest": 16.5,
              "remaining_balance": 250.0
            }
          ]
        }
      ],
      "equity_data": {
        "market_cap": 100.0,
        "share_price": 10.0,
        "volume_avg_30d": 1000.0,
        "pe_ratio": 15.0,
        "dividend_yield": 1.0,
        "beta": 1.0
      }
    },
    "Apple_Inc": {
      "borrower_name": "Apple Inc.",
      "report_date": "2026-02-14T01:37:17.030616",
      "executive_summary": "The borrower Apple Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($148000.0M) but elevated leverage (0.6x).  [Ref: AAPL_10Q_FY25_Q1.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower Apple Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($148000.0M) but elevated leverage (0.6x).  [Ref: AAPL_10Q_FY25_Q1.pdf]",
          "citations": [
            {
              "doc_id": "AAPL_10Q_FY25_Q1.pdf",
              "chunk_id": "a3e0c9f6-1b93-42b2-ab74-3dfc353b5e35",
              "page_number": 6
            }
          ],
          "author_agent": "Writer"
        },
        {
          "title": "Key Risks & Mitigants",
          "content": "Key Risks:\n- CONNECTED PARTY RISK: European Commission via Apple Inc. -> European Commission is High risk.\n- CONNECTED PARTY RISK: Google (Alphabet) via Apple Inc. -> Google (Alphabet) is High risk.\n",
          "citations": [],
          "author_agent": "Risk Officer"
        },
        {
          "title": "Financial Analysis",
          "content": "EBITDA: $148000.0M | Leverage: 0.6x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "financial_ratios": {
        "leverage_ratio": 0.6081081081081081,
        "dscr": 46.25,
        "current_ratio": 1.5,
        "revenue": 448000.0,
        "ebitda": 148000.0,
        "net_income": 88800.0
      },
      "historical_financials": [
        {
          "total_assets": 410000.0,
          "total_liabilities": 310000.0,
          "total_equity": 100000.0,
          "revenue": 448000.0,
          "ebitda": 148000.0,
          "net_income": 88800.0,
          "interest_expense": 3200.0,
          "dscr": 46.25,
          "leverage_ratio": 0.6081081081081081,
          "current_ratio": 1.5,
          "period": "FY2026"
        },
        {
          "total_assets": 377200.0,
          "total_liabilities": 285200.0,
          "total_equity": 92000.0,
          "revenue": 403200.0,
          "ebitda": 130240.0,
          "net_income": 75480.0,
          "interest_expense": 3200.0,
          "dscr": 46.25,
          "leverage_ratio": 0.6081081081081081,
          "current_ratio": 1.5,
          "period": "FY2025"
        },
        {
          "total_assets": 358340.0,
          "total_liabilities": 270940.0,
          "total_equity": 87400.0,
          "revenue": 370944.0,
          "ebitda": 117216.0,
          "net_income": 66422.4,
          "interest_expense": 3200.0,
          "dscr": 46.25,
          "leverage_ratio": 0.6081081081081081,
          "current_ratio": 1.5,
          "period": "FY2024"
        }
      ],
      "dcf_analysis": {
        "free_cash_flow": [
          101010.0,
          105052.80499999998,
          108211.9168,
          110391.71266012496,
          111522.16594766
        ],
        "growth_rate": 0.03,
        "wacc": 0.09,
        "terminal_value": 1914463.8487681635,
        "enterprise_value": 1659606.1949807901,
        "equity_value": 1349606.1949807901,
        "share_price": 185.0
      },
      "risk_score": 75.0,
      "credit_ratings": [
        {
          "agency": "Moody's",
          "rating": "Aaa",
          "outlook": "Stable",
          "date": "2026-02-14"
        },
        {
          "agency": "S&P",
          "rating": "AA+",
          "outlook": "Stable",
          "date": "2026-02-14"
        },
        {
          "agency": "Fitch",
          "rating": "AA+",
          "outlook": "Stable",
          "date": "2026-02-14"
        }
      ],
      "debt_facilities": [
        {
          "facility_type": "Revolving Credit Facility",
          "amount_committed": 10000.0,
          "amount_drawn": 0.0,
          "interest_rate": "SOFR + 0.75%",
          "maturity_date": "2028-09-15",
          "snc_rating": "Pass",
          "drc": 1.0,
          "ltv": 0.0,
          "conviction_score": 0.98,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 0.0,
              "principal": 0.0,
              "interest": 0.0,
              "remaining_balance": 0.0
            },
            {
              "year": 2028,
              "payment_amount": 0.0,
              "principal": 0.0,
              "interest": 0.0,
              "remaining_balance": 0.0
            },
            {
              "year": 2029,
              "payment_amount": 0.0,
              "principal": 0.0,
              "interest": 0.0,
              "remaining_balance": 0.0
            },
            {
              "year": 2030,
              "payment_amount": 0.0,
              "principal": 0.0,
              "interest": 0.0,
              "remaining_balance": 0.0
            },
            {
              "year": 2031,
              "payment_amount": 0.0,
              "principal": 0.0,
              "interest": 0.0,
              "remaining_balance": 0.0
            }
          ]
        },
        {
          "facility_type": "Senior Unsecured Notes (2030)",
          "amount_committed": 2500.0,
          "amount_drawn": 2500.0,
          "interest_rate": "3.25%",
          "maturity_date": "2030-05-11",
          "snc_rating": "Pass",
          "drc": 1.0,
          "ltv": 0.1,
          "conviction_score": 0.95,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 331.25,
              "principal": 250.0,
              "interest": 81.25,
              "remaining_balance": 2250.0
            },
            {
              "year": 2028,
              "payment_amount": 323.12,
              "principal": 250.0,
              "interest": 73.12,
              "remaining_balance": 2000.0
            },
            {
              "year": 2029,
              "payment_amount": 315.0,
              "principal": 250.0,
              "interest": 65.0,
              "remaining_balance": 1750.0
            },
            {
              "year": 2030,
              "payment_amount": 306.88,
              "principal": 250.0,
              "interest": 56.88,
              "remaining_balance": 1500.0
            },
            {
              "year": 2031,
              "payment_amount": 298.75,
              "principal": 250.0,
              "interest": 48.75,
              "remaining_balance": 1250.0
            }
          ]
        },
        {
          "facility_type": "Senior Unsecured Notes (2040)",
          "amount_committed": 1500.0,
          "amount_drawn": 1500.0,
          "interest_rate": "4.10%",
          "maturity_date": "2040-02-28",
          "snc_rating": "Pass",
          "drc": 1.0,
          "ltv": 0.1,
          "conviction_score": 0.92,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 211.5,
              "principal": 150.0,
              "interest": 61.5,
              "remaining_balance": 1350.0
            },
            {
              "year": 2028,
              "payment_amount": 205.35,
              "principal": 150.0,
              "interest": 55.35,
              "remaining_balance": 1200.0
            },
            {
              "year": 2029,
              "payment_amount": 199.2,
              "principal": 150.0,
              "interest": 49.2,
              "remaining_balance": 1050.0
            },
            {
              "year": 2030,
              "payment_amount": 193.05,
              "principal": 150.0,
              "interest": 43.05,
              "remaining_balance": 900.0
            },
            {
              "year": 2031,
              "payment_amount": 186.9,
              "principal": 150.0,
              "interest": 36.9,
              "remaining_balance": 750.0
            }
          ]
        },
        {
          "facility_type": "Term Loan A",
          "amount_committed": 5000.0,
          "amount_drawn": 5000.0,
          "interest_rate": "SOFR + 1.10%",
          "maturity_date": "2027-03-30",
          "snc_rating": "Pass",
          "drc": 1.0,
          "ltv": 0.2,
          "conviction_score": 0.9,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 555.0,
              "principal": 500.0,
              "interest": 55.0,
              "remaining_balance": 4500.0
            },
            {
              "year": 2028,
              "payment_amount": 549.5,
              "principal": 500.0,
              "interest": 49.5,
              "remaining_balance": 4000.0
            },
            {
              "year": 2029,
              "payment_amount": 544.0,
              "principal": 500.0,
              "interest": 44.0,
              "remaining_balance": 3500.0
            },
            {
              "year": 2030,
              "payment_amount": 538.5,
              "principal": 500.0,
              "interest": 38.5,
              "remaining_balance": 3000.0
            },
            {
              "year": 2031,
              "payment_amount": 533.0,
              "principal": 500.0,
              "interest": 33.0,
              "remaining_balance": 2500.0
            }
          ]
        }
      ],
      "equity_data": {
        "market_cap": 3450000.0,
        "share_price": 225.5,
        "volume_avg_30d": 45000000.0,
        "pe_ratio": 31.5,
        "dividend_yield": 0.55,
        "beta": 1.15
      }
    },
    "Alphabet_Inc": {
      "borrower_name": "Alphabet Inc.",
      "report_date": "2026-02-14T01:37:17.038187",
      "executive_summary": "The borrower Alphabet Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($138000.0M) but elevated leverage (0.1x).  [Ref: TechCorp_10K_2025.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower Alphabet Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($138000.0M) but elevated leverage (0.1x).  [Ref: TechCorp_10K_2025.pdf]",
          "citations": [
            {
              "doc_id": "TechCorp_10K_2025.pdf",
              "chunk_id": "7af7d36b-92a2-4742-9d4c-d036c5cd6818",
              "page_number": 3
            }
          ],
          "author_agent": "Writer"
        },
        {
          "title": "Key Risks & Mitigants",
          "content": "Key Risks:\n",
          "citations": [],
          "author_agent": "Risk Officer"
        },
        {
          "title": "Financial Analysis",
          "content": "EBITDA: $138000.0M | Leverage: 0.1x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "financial_ratios": {
        "leverage_ratio": 0.07246376811594203,
        "dscr": 552.0,
        "current_ratio": 1.5,
        "revenue": 395000.0,
        "ebitda": 138000.0,
        "net_income": 82800.0
      },
      "historical_financials": [
        {
          "total_assets": 530000.0,
          "total_liabilities": 150000.0,
          "total_equity": 380000.0,
          "revenue": 395000.0,
          "ebitda": 138000.0,
          "net_income": 82800.0,
          "interest_expense": 250.0,
          "dscr": 552.0,
          "leverage_ratio": 0.07246376811594203,
          "current_ratio": 1.5,
          "period": "FY2026"
        },
        {
          "total_assets": 487600.0,
          "total_liabilities": 138000.0,
          "total_equity": 349600.0,
          "revenue": 355500.0,
          "ebitda": 121440.0,
          "net_income": 70380.0,
          "interest_expense": 250.0,
          "dscr": 552.0,
          "leverage_ratio": 0.07246376811594203,
          "current_ratio": 1.5,
          "period": "FY2025"
        },
        {
          "total_assets": 463220.0,
          "total_liabilities": 131100.0,
          "total_equity": 332120.0,
          "revenue": 327060.0,
          "ebitda": 109296.0,
          "net_income": 61934.4,
          "interest_expense": 250.0,
          "dscr": 552.0,
          "leverage_ratio": 0.07246376811594203,
          "current_ratio": 1.5,
          "period": "FY2024"
        }
      ],
      "dcf_analysis": {
        "free_cash_flow": [
          94185.0,
          97954.64249999999,
          100900.30080000001,
          102932.81315606248,
          103986.88446471
        ],
        "growth_rate": 0.03,
        "wacc": 0.09,
        "terminal_value": 1785108.1833108552,
        "enterprise_value": 1547470.641265872,
        "equity_value": 1397470.641265872,
        "share_price": 185.0
      },
      "risk_score": 75.0,
      "credit_ratings": [
        {
          "agency": "Moody's",
          "rating": "Ba2",
          "outlook": "Stable",
          "date": "2026-02-14"
        },
        {
          "agency": "S&P",
          "rating": "BB",
          "outlook": "Stable",
          "date": "2026-02-14"
        }
      ],
      "debt_facilities": [
        {
          "facility_type": "Revolver",
          "amount_committed": 100.0,
          "amount_drawn": 20.0,
          "interest_rate": "Prime + 1.0%",
          "maturity_date": "2026-01-01",
          "snc_rating": "Pass",
          "drc": 0.8,
          "ltv": 0.5,
          "conviction_score": 0.7,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 2.2,
              "principal": 2.0,
              "interest": 0.2,
              "remaining_balance": 18.0
            },
            {
              "year": 2028,
              "payment_amount": 2.18,
              "principal": 2.0,
              "interest": 0.18,
              "remaining_balance": 16.0
            },
            {
              "year": 2029,
              "payment_amount": 2.16,
              "principal": 2.0,
              "interest": 0.16,
              "remaining_balance": 14.0
            },
            {
              "year": 2030,
              "payment_amount": 2.14,
              "principal": 2.0,
              "interest": 0.14,
              "remaining_balance": 12.0
            },
            {
              "year": 2031,
              "payment_amount": 2.12,
              "principal": 2.0,
              "interest": 0.12,
              "remaining_balance": 10.0
            }
          ]
        },
        {
          "facility_type": "Term Loan",
          "amount_committed": 500.0,
          "amount_drawn": 500.0,
          "interest_rate": "5.50%",
          "maturity_date": "2029-01-01",
          "snc_rating": "Pass",
          "drc": 0.75,
          "ltv": 0.6,
          "conviction_score": 0.75,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 77.5,
              "principal": 50.0,
              "interest": 27.5,
              "remaining_balance": 450.0
            },
            {
              "year": 2028,
              "payment_amount": 74.75,
              "principal": 50.0,
              "interest": 24.75,
              "remaining_balance": 400.0
            },
            {
              "year": 2029,
              "payment_amount": 72.0,
              "principal": 50.0,
              "interest": 22.0,
              "remaining_balance": 350.0
            },
            {
              "year": 2030,
              "payment_amount": 69.25,
              "principal": 50.0,
              "interest": 19.25,
              "remaining_balance": 300.0
            },
            {
              "year": 2031,
              "payment_amount": 66.5,
              "principal": 50.0,
              "interest": 16.5,
              "remaining_balance": 250.0
            }
          ]
        }
      ],
      "equity_data": {
        "market_cap": 100.0,
        "share_price": 10.0,
        "volume_avg_30d": 1000.0,
        "pe_ratio": 15.0,
        "dividend_yield": 1.0,
        "beta": 1.0
      }
    },
    "NVIDIA_Corp": {
      "borrower_name": "NVIDIA Corp",
      "report_date": "2026-02-14T01:37:17.037134",
      "executive_summary": "The borrower NVIDIA Corp presents a mixed credit profile. Financial performance shows strong EBITDA ($92000.0M) but elevated leverage (0.1x).  [Ref: TechCorp_10K_2025.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower NVIDIA Corp presents a mixed credit profile. Financial performance shows strong EBITDA ($92000.0M) but elevated leverage (0.1x).  [Ref: TechCorp_10K_2025.pdf]",
          "citations": [
            {
              "doc_id": "TechCorp_10K_2025.pdf",
              "chunk_id": "0d27bd24-d3f5-4fe7-9032-90617a2338ab",
              "page_number": 3
            }
          ],
          "author_agent": "Writer"
        },
        {
          "title": "Key Risks & Mitigants",
          "content": "Key Risks:\n",
          "citations": [],
          "author_agent": "Risk Officer"
        },
        {
          "title": "Financial Analysis",
          "content": "EBITDA: $92000.0M | Leverage: 0.1x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "financial_ratios": {
        "leverage_ratio": 0.08152173913043478,
        "dscr": 418.1818181818182,
        "current_ratio": 1.5,
        "revenue": 140000.0,
        "ebitda": 92000.0,
        "net_income": 55200.0
      },
      "historical_financials": [
        {
          "total_assets": 130000.0,
          "total_liabilities": 30000.0,
          "total_equity": 100000.0,
          "revenue": 140000.0,
          "ebitda": 92000.0,
          "net_income": 55200.0,
          "interest_expense": 220.0,
          "dscr": 418.1818181818182,
          "leverage_ratio": 0.08152173913043478,
          "current_ratio": 1.5,
          "period": "FY2026"
        },
        {
          "total_assets": 119600.0,
          "total_liabilities": 27600.0,
          "total_equity": 92000.0,
          "revenue": 126000.0,
          "ebitda": 80960.0,
          "net_income": 46920.0,
          "interest_expense": 220.0,
          "dscr": 418.1818181818182,
          "leverage_ratio": 0.08152173913043478,
          "current_ratio": 1.5,
          "period": "FY2025"
        },
        {
          "total_assets": 113620.0,
          "total_liabilities": 26220.0,
          "total_equity": 87400.0,
          "revenue": 115920.0,
          "ebitda": 72864.0,
          "net_income": 41289.6,
          "interest_expense": 220.0,
          "dscr": 418.1818181818182,
          "leverage_ratio": 0.08152173913043478,
          "current_ratio": 1.5,
          "period": "FY2024"
        }
      ],
      "dcf_analysis": {
        "free_cash_flow": [
          62790.0,
          65303.09499999999,
          67266.86720000001,
          68621.87543737498,
          69324.58964314
        ],
        "growth_rate": 0.03,
        "wacc": 0.09,
        "terminal_value": 1190072.1222072367,
        "enterprise_value": 1031647.094177248,
        "equity_value": 1001647.094177248,
        "share_price": 185.0
      },
      "risk_score": 75.0,
      "credit_ratings": [
        {
          "agency": "Moody's",
          "rating": "Ba2",
          "outlook": "Stable",
          "date": "2026-02-14"
        },
        {
          "agency": "S&P",
          "rating": "BB",
          "outlook": "Stable",
          "date": "2026-02-14"
        }
      ],
      "debt_facilities": [
        {
          "facility_type": "Revolver",
          "amount_committed": 100.0,
          "amount_drawn": 20.0,
          "interest_rate": "Prime + 1.0%",
          "maturity_date": "2026-01-01",
          "snc_rating": "Pass",
          "drc": 0.8,
          "ltv": 0.5,
          "conviction_score": 0.7,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 2.2,
              "principal": 2.0,
              "interest": 0.2,
              "remaining_balance": 18.0
            },
            {
              "year": 2028,
              "payment_amount": 2.18,
              "principal": 2.0,
              "interest": 0.18,
              "remaining_balance": 16.0
            },
            {
              "year": 2029,
              "payment_amount": 2.16,
              "principal": 2.0,
              "interest": 0.16,
              "remaining_balance": 14.0
            },
            {
              "year": 2030,
              "payment_amount": 2.14,
              "principal": 2.0,
              "interest": 0.14,
              "remaining_balance": 12.0
            },
            {
              "year": 2031,
              "payment_amount": 2.12,
              "principal": 2.0,
              "interest": 0.12,
              "remaining_balance": 10.0
            }
          ]
        },
        {
          "facility_type": "Term Loan",
          "amount_committed": 500.0,
          "amount_drawn": 500.0,
          "interest_rate": "5.50%",
          "maturity_date": "2029-01-01",
          "snc_rating": "Pass",
          "drc": 0.75,
          "ltv": 0.6,
          "conviction_score": 0.75,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 77.5,
              "principal": 50.0,
              "interest": 27.5,
              "remaining_balance": 450.0
            },
            {
              "year": 2028,
              "payment_amount": 74.75,
              "principal": 50.0,
              "interest": 24.75,
              "remaining_balance": 400.0
            },
            {
              "year": 2029,
              "payment_amount": 72.0,
              "principal": 50.0,
              "interest": 22.0,
              "remaining_balance": 350.0
            },
            {
              "year": 2030,
              "payment_amount": 69.25,
              "principal": 50.0,
              "interest": 19.25,
              "remaining_balance": 300.0
            },
            {
              "year": 2031,
              "payment_amount": 66.5,
              "principal": 50.0,
              "interest": 16.5,
              "remaining_balance": 250.0
            }
          ]
        }
      ],
      "equity_data": {
        "market_cap": 100.0,
        "share_price": 10.0,
        "volume_avg_30d": 1000.0,
        "pe_ratio": 15.0,
        "dividend_yield": 1.0,
        "beta": 1.0
      }
    },
    "Bank_of_America": {
      "borrower_name": "Bank of America",
      "report_date": "2026-02-14T01:37:17.044911",
      "executive_summary": "The borrower Bank of America presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: Generic_Borrower_Profile.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower Bank of America presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: Generic_Borrower_Profile.pdf]",
          "citations": [
            {
              "doc_id": "Generic_Borrower_Profile.pdf",
              "chunk_id": "9ad4131e-4c44-42ad-8837-0233c09c775a",
              "page_number": 1
            }
          ],
          "author_agent": "Writer"
        },
        {
          "title": "Key Risks & Mitigants",
          "content": "Key Risks:\n- HIGH: Leverage ratio exceeds 4.0x policy limit.\n",
          "citations": [],
          "author_agent": "Risk Officer"
        },
        {
          "title": "Financial Analysis",
          "content": "EBITDA: $5.0M | Leverage: 10.0x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "financial_ratios": {
        "leverage_ratio": 10.0,
        "dscr": 5.0,
        "current_ratio": 2.0,
        "revenue": 20.0,
        "ebitda": 5.0,
        "net_income": 2.0
      },
      "historical_financials": [
        {
          "total_assets": 100.0,
          "total_liabilities": 50.0,
          "total_equity": 50.0,
          "revenue": 20.0,
          "ebitda": 5.0,
          "net_income": 2.0,
          "interest_expense": 1.0,
          "dscr": 5.0,
          "leverage_ratio": 10.0,
          "current_ratio": 2.0,
          "period": "FY2025"
        },
        {
          "total_assets": 92.0,
          "total_liabilities": 46.0,
          "total_equity": 46.0,
          "revenue": 18.0,
          "ebitda": 4.4,
          "net_income": 1.7,
          "interest_expense": 1.0,
          "dscr": 5.0,
          "leverage_ratio": 10.0,
          "current_ratio": 2.0,
          "period": "FY2024"
        },
        {
          "total_assets": 87.39999999999999,
          "total_liabilities": 43.699999999999996,
          "total_equity": 43.699999999999996,
          "revenue": 16.560000000000002,
          "ebitda": 3.9600000000000004,
          "net_income": 1.496,
          "interest_expense": 1.0,
          "dscr": 5.0,
          "leverage_ratio": 10.0,
          "current_ratio": 2.0,
          "period": "FY2023"
        }
      ],
      "dcf_analysis": {
        "free_cash_flow": [
          3.4125,
          3.5490812499999995,
          3.6558080000000004,
          3.729449752031249,
          3.767640741475
        ],
        "growth_rate": 0.03,
        "wacc": 0.09,
        "terminal_value": 64.67783272865418,
        "enterprise_value": 56.06777685745914,
        "equity_value": 6.067776857459137,
        "share_price": 185.0
      },
      "risk_score": 45.0,
      "credit_ratings": [
        {
          "agency": "Moody's",
          "rating": "Ba2",
          "outlook": "Stable",
          "date": "2026-02-14"
        },
        {
          "agency": "S&P",
          "rating": "BB",
          "outlook": "Stable",
          "date": "2026-02-14"
        }
      ],
      "debt_facilities": [
        {
          "facility_type": "Revolver",
          "amount_committed": 100.0,
          "amount_drawn": 20.0,
          "interest_rate": "Prime + 1.0%",
          "maturity_date": "2026-01-01",
          "snc_rating": "Pass",
          "drc": 0.8,
          "ltv": 0.5,
          "conviction_score": 0.7,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 2.2,
              "principal": 2.0,
              "interest": 0.2,
              "remaining_balance": 18.0
            },
            {
              "year": 2028,
              "payment_amount": 2.18,
              "principal": 2.0,
              "interest": 0.18,
              "remaining_balance": 16.0
            },
            {
              "year": 2029,
              "payment_amount": 2.16,
              "principal": 2.0,
              "interest": 0.16,
              "remaining_balance": 14.0
            },
            {
              "year": 2030,
              "payment_amount": 2.14,
              "principal": 2.0,
              "interest": 0.14,
              "remaining_balance": 12.0
            },
            {
              "year": 2031,
              "payment_amount": 2.12,
              "principal": 2.0,
              "interest": 0.12,
              "remaining_balance": 10.0
            }
          ]
        },
        {
          "facility_type": "Term Loan",
          "amount_committed": 500.0,
          "amount_drawn": 500.0,
          "interest_rate": "5.50%",
          "maturity_date": "2029-01-01",
          "snc_rating": "Pass",
          "drc": 0.75,
          "ltv": 0.6,
          "conviction_score": 0.75,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 77.5,
              "principal": 50.0,
              "interest": 27.5,
              "remaining_balance": 450.0
            },
            {
              "year": 2028,
              "payment_amount": 74.75,
              "principal": 50.0,
              "interest": 24.75,
              "remaining_balance": 400.0
            },
            {
              "year": 2029,
              "payment_amount": 72.0,
              "principal": 50.0,
              "interest": 22.0,
              "remaining_balance": 350.0
            },
            {
              "year": 2030,
              "payment_amount": 69.25,
              "principal": 50.0,
              "interest": 19.25,
              "remaining_balance": 300.0
            },
            {
              "year": 2031,
              "payment_amount": 66.5,
              "principal": 50.0,
              "interest": 16.5,
              "remaining_balance": 250.0
            }
          ]
        }
      ],
      "equity_data": {
        "market_cap": 100.0,
        "share_price": 10.0,
        "volume_avg_30d": 1000.0,
        "pe_ratio": 15.0,
        "dividend_yield": 1.0,
        "beta": 1.0
      }
    },
    "JPMorgan_Chase": {
      "borrower_name": "JPMorgan Chase",
      "report_date": "2026-02-14T01:37:17.043144",
      "executive_summary": "The borrower JPMorgan Chase presents a mixed credit profile. Financial performance shows strong EBITDA ($105000.0M) but elevated leverage (10.2x).  [Ref: JPM_Earnings_Release_4Q24.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower JPMorgan Chase presents a mixed credit profile. Financial performance shows strong EBITDA ($105000.0M) but elevated leverage (10.2x).  [Ref: JPM_Earnings_Release_4Q24.pdf]",
          "citations": [
            {
              "doc_id": "JPM_Earnings_Release_4Q24.pdf",
              "chunk_id": "00c69495-7115-48eb-bc19-df327eeb5bcd",
              "page_number": 1
            }
          ],
          "author_agent": "Writer"
        },
        {
          "title": "Key Risks & Mitigants",
          "content": "Key Risks:\n- HIGH: Leverage ratio exceeds 4.0x policy limit.\n",
          "citations": [],
          "author_agent": "Risk Officer"
        },
        {
          "title": "Financial Analysis",
          "content": "EBITDA: $105000.0M | Leverage: 10.2x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "financial_ratios": {
        "leverage_ratio": 10.25,
        "dscr": 999.0,
        "current_ratio": 1.1,
        "revenue": 190000.0,
        "ebitda": 105000.0,
        "net_income": 65000.0
      },
      "historical_financials": [
        {
          "total_assets": 4500000.0,
          "total_liabilities": 4100000.0,
          "total_equity": 400000.0,
          "revenue": 190000.0,
          "ebitda": 105000.0,
          "net_income": 65000.0,
          "interest_expense": 1.0,
          "dscr": 999.0,
          "leverage_ratio": 10.25,
          "current_ratio": 1.1,
          "period": "FY2026 (Proj)"
        },
        {
          "total_assets": 4140000.0,
          "total_liabilities": 3772000.0,
          "total_equity": 368000.0,
          "revenue": 171000.0,
          "ebitda": 92400.0,
          "net_income": 55250.0,
          "interest_expense": 1.0,
          "dscr": 999.0,
          "leverage_ratio": 10.25,
          "current_ratio": 1.1,
          "period": "FY2025"
        },
        {
          "total_assets": 3933000.0,
          "total_liabilities": 3583400.0,
          "total_equity": 349600.0,
          "revenue": 157320.0,
          "ebitda": 83160.0,
          "net_income": 48620.0,
          "interest_expense": 1.0,
          "dscr": 999.0,
          "leverage_ratio": 10.25,
          "current_ratio": 1.1,
          "period": "FY2024"
        }
      ],
      "dcf_analysis": {
        "free_cash_flow": [
          71662.5,
          74530.70624999999,
          76771.96800000001,
          78318.44479265623,
          79120.455570975
        ],
        "growth_rate": 0.03,
        "wacc": 0.09,
        "terminal_value": 1358234.4873017375,
        "enterprise_value": 1177423.3140066417,
        "equity_value": -2922576.6859933585,
        "share_price": -2922576.6859933585
      },
      "risk_score": 45.0,
      "credit_ratings": [
        {
          "agency": "Moody's",
          "rating": "A1",
          "outlook": "Stable",
          "date": "2026-02-14"
        },
        {
          "agency": "S&P",
          "rating": "A-",
          "outlook": "Stable",
          "date": "2026-02-14"
        },
        {
          "agency": "Fitch",
          "rating": "AA-",
          "outlook": "Stable",
          "date": "2026-02-14"
        }
      ],
      "debt_facilities": [
        {
          "facility_type": "Revolver",
          "amount_committed": 100.0,
          "amount_drawn": 20.0,
          "interest_rate": "Prime + 1.0%",
          "maturity_date": "2026-01-01",
          "snc_rating": "Pass",
          "drc": 0.8,
          "ltv": 0.5,
          "conviction_score": 0.7,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 2.2,
              "principal": 2.0,
              "interest": 0.2,
              "remaining_balance": 18.0
            },
            {
              "year": 2028,
              "payment_amount": 2.18,
              "principal": 2.0,
              "interest": 0.18,
              "remaining_balance": 16.0
            },
            {
              "year": 2029,
              "payment_amount": 2.16,
              "principal": 2.0,
              "interest": 0.16,
              "remaining_balance": 14.0
            },
            {
              "year": 2030,
              "payment_amount": 2.14,
              "principal": 2.0,
              "interest": 0.14,
              "remaining_balance": 12.0
            },
            {
              "year": 2031,
              "payment_amount": 2.12,
              "principal": 2.0,
              "interest": 0.12,
              "remaining_balance": 10.0
            }
          ]
        },
        {
          "facility_type": "Term Loan",
          "amount_committed": 500.0,
          "amount_drawn": 500.0,
          "interest_rate": "5.50%",
          "maturity_date": "2029-01-01",
          "snc_rating": "Pass",
          "drc": 0.75,
          "ltv": 0.6,
          "conviction_score": 0.75,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 77.5,
              "principal": 50.0,
              "interest": 27.5,
              "remaining_balance": 450.0
            },
            {
              "year": 2028,
              "payment_amount": 74.75,
              "principal": 50.0,
              "interest": 24.75,
              "remaining_balance": 400.0
            },
            {
              "year": 2029,
              "payment_amount": 72.0,
              "principal": 50.0,
              "interest": 22.0,
              "remaining_balance": 350.0
            },
            {
              "year": 2030,
              "payment_amount": 69.25,
              "principal": 50.0,
              "interest": 19.25,
              "remaining_balance": 300.0
            },
            {
              "year": 2031,
              "payment_amount": 66.5,
              "principal": 50.0,
              "interest": 16.5,
              "remaining_balance": 250.0
            }
          ]
        }
      ],
      "equity_data": {
        "market_cap": 580000.0,
        "share_price": 205.1,
        "volume_avg_30d": 9500000.0,
        "pe_ratio": 11.8,
        "dividend_yield": 2.3,
        "beta": 1.05
      }
    },
    "TechCorp_Inc": {
      "borrower_name": "TechCorp Inc.",
      "report_date": "2026-02-13T02:55:10.459724",
      "executive_summary": "The borrower TechCorp Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($350.0M) but elevated leverage (8.6x).  [Ref: TechCorp_10K_2025.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower TechCorp Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($350.0M) but elevated leverage (8.6x).  [Ref: TechCorp_10K_2025.pdf]",
          "citations": [
            {
              "doc_id": "TechCorp_10K_2025.pdf",
              "chunk_id": "bf57b34d-cbc1-4ce5-a323-c928d2214657",
              "page_number": 3
            }
          ],
          "author_agent": "Writer"
        },
        {
          "title": "Key Risks & Mitigants",
          "content": "Key Risks:\n- HIGH: Leverage ratio exceeds 4.0x policy limit.\n",
          "citations": [],
          "author_agent": "Risk Officer"
        },
        {
          "title": "Financial Analysis",
          "content": "EBITDA: $350.0M | Leverage: 8.6x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "financial_ratios": {
        "leverage_ratio": 8.571428571428571,
        "dscr": 7.0,
        "current_ratio": 1.5,
        "revenue": 1200.0,
        "ebitda": 350.0,
        "net_income": 150.0
      },
      "historical_financials": [
        {
          "total_assets": 5000.0,
          "total_liabilities": 3000.0,
          "total_equity": 2000.0,
          "revenue": 1200.0,
          "ebitda": 350.0,
          "net_income": 150.0,
          "interest_expense": 50.0,
          "dscr": 7.0,
          "leverage_ratio": 8.571428571428571,
          "current_ratio": 1.5,
          "period": "FY2025 Mock"
        },
        {
          "total_assets": 4600.0,
          "total_liabilities": 2760.0,
          "total_equity": 1840.0,
          "revenue": 1080.0,
          "ebitda": 308.0,
          "net_income": 127.5,
          "interest_expense": 50.0,
          "dscr": 7.0,
          "leverage_ratio": 8.571428571428571,
          "current_ratio": 1.5,
          "period": "FY2024"
        },
        {
          "total_assets": 4370.0,
          "total_liabilities": 2622.0,
          "total_equity": 1748.0,
          "revenue": 993.6,
          "ebitda": 277.2,
          "net_income": 112.2,
          "interest_expense": 50.0,
          "dscr": 7.0,
          "leverage_ratio": 8.571428571428571,
          "current_ratio": 1.5,
          "period": "FY2023"
        }
      ],
      "dcf_analysis": {
        "free_cash_flow": [
          238.875,
          248.43568749999994,
          255.90656,
          261.0614826421874,
          263.73485190325
        ],
        "growth_rate": 0.03,
        "wacc": 0.09,
        "terminal_value": 4527.448291005791,
        "enterprise_value": 3924.7443800221386,
        "equity_value": 924.7443800221386,
        "share_price": 185.0
      },
      "risk_score": 45.0,
      "credit_ratings": [
        {
          "agency": "Moody's",
          "rating": "B2",
          "outlook": "Negative",
          "date": "2026-02-13"
        },
        {
          "agency": "S&P",
          "rating": "B",
          "outlook": "Watch",
          "date": "2026-02-13"
        }
      ],
      "debt_facilities": [
        {
          "facility_type": "Revolver",
          "amount_committed": 500.0,
          "amount_drawn": 350.0,
          "interest_rate": "SOFR + 3.50%",
          "maturity_date": "2025-12-31",
          "snc_rating": "Special Mention",
          "drc": 0.4,
          "ltv": 0.7,
          "conviction_score": 0.45
        },
        {
          "facility_type": "Term Loan B",
          "amount_committed": 1200.0,
          "amount_drawn": 1200.0,
          "interest_rate": "SOFR + 4.75%",
          "maturity_date": "2028-06-30",
          "snc_rating": "Substandard",
          "drc": 0.35,
          "ltv": 0.85,
          "conviction_score": 0.3
        },
        {
          "facility_type": "Mezzanine Debt",
          "amount_committed": 300.0,
          "amount_drawn": 300.0,
          "interest_rate": "12.00% PIK",
          "maturity_date": "2029-06-30",
          "snc_rating": "Doubtful",
          "drc": 0.1,
          "ltv": 0.95,
          "conviction_score": 0.15
        }
      ],
      "equity_data": {
        "market_cap": 1200.0,
        "share_price": 15.25,
        "volume_avg_30d": 150000.0,
        "pe_ratio": 18.5,
        "dividend_yield": 0.0,
        "beta": 1.85
      }
    },
    "Microsoft_Corp": {
      "borrower_name": "Microsoft Corp",
      "report_date": "2026-02-14T01:37:17.035685",
      "executive_summary": "The borrower Microsoft Corp presents a mixed credit profile. Financial performance shows strong EBITDA ($165000.0M) but elevated leverage (0.2x).  [Ref: TechCorp_10K_2025.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower Microsoft Corp presents a mixed credit profile. Financial performance shows strong EBITDA ($165000.0M) but elevated leverage (0.2x).  [Ref: TechCorp_10K_2025.pdf]",
          "citations": [
            {
              "doc_id": "TechCorp_10K_2025.pdf",
              "chunk_id": "7c40c6d1-8418-4fdc-b98d-24624c871174",
              "page_number": 3
            }
          ],
          "author_agent": "Writer"
        },
        {
          "title": "Key Risks & Mitigants",
          "content": "Key Risks:\n",
          "citations": [],
          "author_agent": "Risk Officer"
        },
        {
          "title": "Financial Analysis",
          "content": "EBITDA: $165000.0M | Leverage: 0.2x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "financial_ratios": {
        "leverage_ratio": 0.24242424242424243,
        "dscr": 110.0,
        "current_ratio": 1.5,
        "revenue": 320000.0,
        "ebitda": 165000.0,
        "net_income": 99000.0
      },
      "historical_financials": [
        {
          "total_assets": 560000.0,
          "total_liabilities": 260000.0,
          "total_equity": 300000.0,
          "revenue": 320000.0,
          "ebitda": 165000.0,
          "net_income": 99000.0,
          "interest_expense": 1500.0,
          "dscr": 110.0,
          "leverage_ratio": 0.24242424242424243,
          "current_ratio": 1.5,
          "period": "FY2026"
        },
        {
          "total_assets": 515200.0,
          "total_liabilities": 239200.0,
          "total_equity": 276000.0,
          "revenue": 288000.0,
          "ebitda": 145200.0,
          "net_income": 84150.0,
          "interest_expense": 1500.0,
          "dscr": 110.0,
          "leverage_ratio": 0.24242424242424243,
          "current_ratio": 1.5,
          "period": "FY2025"
        },
        {
          "total_assets": 489440.0,
          "total_liabilities": 227240.0,
          "total_equity": 262200.0,
          "revenue": 264960.0,
          "ebitda": 130680.0,
          "net_income": 74052.0,
          "interest_expense": 1500.0,
          "dscr": 110.0,
          "leverage_ratio": 0.24242424242424243,
          "current_ratio": 1.5,
          "period": "FY2024"
        }
      ],
      "dcf_analysis": {
        "free_cash_flow": [
          112612.5,
          117119.68124999998,
          120641.664,
          123071.84181703122,
          124332.144468675
        ],
        "growth_rate": 0.03,
        "wacc": 0.09,
        "terminal_value": 2134368.4800455878,
        "enterprise_value": 1850236.6362961512,
        "equity_value": 1590236.6362961512,
        "share_price": 185.0
      },
      "risk_score": 75.0,
      "credit_ratings": [
        {
          "agency": "Moody's",
          "rating": "Ba2",
          "outlook": "Stable",
          "date": "2026-02-14"
        },
        {
          "agency": "S&P",
          "rating": "BB",
          "outlook": "Stable",
          "date": "2026-02-14"
        }
      ],
      "debt_facilities": [
        {
          "facility_type": "Revolver",
          "amount_committed": 100.0,
          "amount_drawn": 20.0,
          "interest_rate": "Prime + 1.0%",
          "maturity_date": "2026-01-01",
          "snc_rating": "Pass",
          "drc": 0.8,
          "ltv": 0.5,
          "conviction_score": 0.7,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 2.2,
              "principal": 2.0,
              "interest": 0.2,
              "remaining_balance": 18.0
            },
            {
              "year": 2028,
              "payment_amount": 2.18,
              "principal": 2.0,
              "interest": 0.18,
              "remaining_balance": 16.0
            },
            {
              "year": 2029,
              "payment_amount": 2.16,
              "principal": 2.0,
              "interest": 0.16,
              "remaining_balance": 14.0
            },
            {
              "year": 2030,
              "payment_amount": 2.14,
              "principal": 2.0,
              "interest": 0.14,
              "remaining_balance": 12.0
            },
            {
              "year": 2031,
              "payment_amount": 2.12,
              "principal": 2.0,
              "interest": 0.12,
              "remaining_balance": 10.0
            }
          ]
        },
        {
          "facility_type": "Term Loan",
          "amount_committed": 500.0,
          "amount_drawn": 500.0,
          "interest_rate": "5.50%",
          "maturity_date": "2029-01-01",
          "snc_rating": "Pass",
          "drc": 0.75,
          "ltv": 0.6,
          "conviction_score": 0.75,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 77.5,
              "principal": 50.0,
              "interest": 27.5,
              "remaining_balance": 450.0
            },
            {
              "year": 2028,
              "payment_amount": 74.75,
              "principal": 50.0,
              "interest": 24.75,
              "remaining_balance": 400.0
            },
            {
              "year": 2029,
              "payment_amount": 72.0,
              "principal": 50.0,
              "interest": 22.0,
              "remaining_balance": 350.0
            },
            {
              "year": 2030,
              "payment_amount": 69.25,
              "principal": 50.0,
              "interest": 19.25,
              "remaining_balance": 300.0
            },
            {
              "year": 2031,
              "payment_amount": 66.5,
              "principal": 50.0,
              "interest": 16.5,
              "remaining_balance": 250.0
            }
          ]
        }
      ],
      "equity_data": {
        "market_cap": 100.0,
        "share_price": 10.0,
        "volume_avg_30d": 1000.0,
        "pe_ratio": 15.0,
        "dividend_yield": 1.0,
        "beta": 1.0
      }
    },
    "Goldman_Sachs": {
      "borrower_name": "Goldman Sachs",
      "report_date": "2026-02-14T01:37:17.044032",
      "executive_summary": "The borrower Goldman Sachs presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: Generic_Borrower_Profile.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower Goldman Sachs presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: Generic_Borrower_Profile.pdf]",
          "citations": [
            {
              "doc_id": "Generic_Borrower_Profile.pdf",
              "chunk_id": "4edd84ac-3669-4f36-9994-7c917ed01d2a",
              "page_number": 1
            }
          ],
          "author_agent": "Writer"
        },
        {
          "title": "Key Risks & Mitigants",
          "content": "Key Risks:\n- HIGH: Leverage ratio exceeds 4.0x policy limit.\n",
          "citations": [],
          "author_agent": "Risk Officer"
        },
        {
          "title": "Financial Analysis",
          "content": "EBITDA: $5.0M | Leverage: 10.0x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "financial_ratios": {
        "leverage_ratio": 10.0,
        "dscr": 5.0,
        "current_ratio": 2.0,
        "revenue": 20.0,
        "ebitda": 5.0,
        "net_income": 2.0
      },
      "historical_financials": [
        {
          "total_assets": 100.0,
          "total_liabilities": 50.0,
          "total_equity": 50.0,
          "revenue": 20.0,
          "ebitda": 5.0,
          "net_income": 2.0,
          "interest_expense": 1.0,
          "dscr": 5.0,
          "leverage_ratio": 10.0,
          "current_ratio": 2.0,
          "period": "FY2025"
        },
        {
          "total_assets": 92.0,
          "total_liabilities": 46.0,
          "total_equity": 46.0,
          "revenue": 18.0,
          "ebitda": 4.4,
          "net_income": 1.7,
          "interest_expense": 1.0,
          "dscr": 5.0,
          "leverage_ratio": 10.0,
          "current_ratio": 2.0,
          "period": "FY2024"
        },
        {
          "total_assets": 87.39999999999999,
          "total_liabilities": 43.699999999999996,
          "total_equity": 43.699999999999996,
          "revenue": 16.560000000000002,
          "ebitda": 3.9600000000000004,
          "net_income": 1.496,
          "interest_expense": 1.0,
          "dscr": 5.0,
          "leverage_ratio": 10.0,
          "current_ratio": 2.0,
          "period": "FY2023"
        }
      ],
      "dcf_analysis": {
        "free_cash_flow": [
          3.4125,
          3.5490812499999995,
          3.6558080000000004,
          3.729449752031249,
          3.767640741475
        ],
        "growth_rate": 0.03,
        "wacc": 0.09,
        "terminal_value": 64.67783272865418,
        "enterprise_value": 56.06777685745914,
        "equity_value": 6.067776857459137,
        "share_price": 185.0
      },
      "risk_score": 45.0,
      "credit_ratings": [
        {
          "agency": "Moody's",
          "rating": "Ba2",
          "outlook": "Stable",
          "date": "2026-02-14"
        },
        {
          "agency": "S&P",
          "rating": "BB",
          "outlook": "Stable",
          "date": "2026-02-14"
        }
      ],
      "debt_facilities": [
        {
          "facility_type": "Revolver",
          "amount_committed": 100.0,
          "amount_drawn": 20.0,
          "interest_rate": "Prime + 1.0%",
          "maturity_date": "2026-01-01",
          "snc_rating": "Pass",
          "drc": 0.8,
          "ltv": 0.5,
          "conviction_score": 0.7,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 2.2,
              "principal": 2.0,
              "interest": 0.2,
              "remaining_balance": 18.0
            },
            {
              "year": 2028,
              "payment_amount": 2.18,
              "principal": 2.0,
              "interest": 0.18,
              "remaining_balance": 16.0
            },
            {
              "year": 2029,
              "payment_amount": 2.16,
              "principal": 2.0,
              "interest": 0.16,
              "remaining_balance": 14.0
            },
            {
              "year": 2030,
              "payment_amount": 2.14,
              "principal": 2.0,
              "interest": 0.14,
              "remaining_balance": 12.0
            },
            {
              "year": 2031,
              "payment_amount": 2.12,
              "principal": 2.0,
              "interest": 0.12,
              "remaining_balance": 10.0
            }
          ]
        },
        {
          "facility_type": "Term Loan",
          "amount_committed": 500.0,
          "amount_drawn": 500.0,
          "interest_rate": "5.50%",
          "maturity_date": "2029-01-01",
          "snc_rating": "Pass",
          "drc": 0.75,
          "ltv": 0.6,
          "conviction_score": 0.75,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 77.5,
              "principal": 50.0,
              "interest": 27.5,
              "remaining_balance": 450.0
            },
            {
              "year": 2028,
              "payment_amount": 74.75,
              "principal": 50.0,
              "interest": 24.75,
              "remaining_balance": 400.0
            },
            {
              "year": 2029,
              "payment_amount": 72.0,
              "principal": 50.0,
              "interest": 22.0,
              "remaining_balance": 350.0
            },
            {
              "year": 2030,
              "payment_amount": 69.25,
              "principal": 50.0,
              "interest": 19.25,
              "remaining_balance": 300.0
            },
            {
              "year": 2031,
              "payment_amount": 66.5,
              "principal": 50.0,
              "interest": 16.5,
              "remaining_balance": 250.0
            }
          ]
        }
      ],
      "equity_data": {
        "market_cap": 100.0,
        "share_price": 10.0,
        "volume_avg_30d": 1000.0,
        "pe_ratio": 15.0,
        "dividend_yield": 1.0,
        "beta": 1.0
      }
    },
    "Tesla_Inc": {
      "borrower_name": "Tesla Inc.",
      "report_date": "2026-02-14T01:37:17.040466",
      "executive_summary": "The borrower Tesla Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($32000.0M) but elevated leverage (0.2x).  [Ref: TSLA_10Q_FY24_Q3.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower Tesla Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($32000.0M) but elevated leverage (0.2x).  [Ref: TSLA_10Q_FY24_Q3.pdf]",
          "citations": [
            {
              "doc_id": "TSLA_10Q_FY24_Q3.pdf",
              "chunk_id": "346e78d3-9cb8-49e7-a30c-94353f94f728",
              "page_number": 8
            }
          ],
          "author_agent": "Writer"
        },
        {
          "title": "Key Risks & Mitigants",
          "content": "Key Risks:\n- CONNECTED PARTY RISK: xAI via Tesla Inc. -> xAI is High risk.\n- CONNECTED PARTY RISK: X (Twitter) via Tesla Inc. -> X (Twitter) is High risk.\n",
          "citations": [],
          "author_agent": "Risk Officer"
        },
        {
          "title": "Financial Analysis",
          "content": "EBITDA: $32000.0M | Leverage: 0.2x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "financial_ratios": {
        "leverage_ratio": 0.21875,
        "dscr": 106.66666666666667,
        "current_ratio": 1.5,
        "revenue": 160000.0,
        "ebitda": 32000.0,
        "net_income": 19200.0
      },
      "historical_financials": [
        {
          "total_assets": 180000.0,
          "total_liabilities": 70000.0,
          "total_equity": 110000.0,
          "revenue": 160000.0,
          "ebitda": 32000.0,
          "net_income": 19200.0,
          "interest_expense": 300.0,
          "dscr": 106.66666666666667,
          "leverage_ratio": 0.21875,
          "current_ratio": 1.5,
          "period": "FY2026"
        },
        {
          "total_assets": 165600.0,
          "total_liabilities": 64400.0,
          "total_equity": 101200.0,
          "revenue": 144000.0,
          "ebitda": 28160.0,
          "net_income": 16320.0,
          "interest_expense": 300.0,
          "dscr": 106.66666666666667,
          "leverage_ratio": 0.21875,
          "current_ratio": 1.5,
          "period": "FY2025"
        },
        {
          "total_assets": 157320.0,
          "total_liabilities": 61180.0,
          "total_equity": 96140.0,
          "revenue": 132480.0,
          "ebitda": 25344.0,
          "net_income": 14361.6,
          "interest_expense": 300.0,
          "dscr": 106.66666666666667,
          "leverage_ratio": 0.21875,
          "current_ratio": 1.5,
          "period": "FY2024"
        }
      ],
      "dcf_analysis": {
        "free_cash_flow": [
          21840.0,
          22714.119999999995,
          23397.1712,
          23868.478412999993,
          24112.90074544
        ],
        "growth_rate": 0.03,
        "wacc": 0.09,
        "terminal_value": 413938.1294633867,
        "enterprise_value": 358833.7718877384,
        "equity_value": 288833.7718877384,
        "share_price": 185.0
      },
      "risk_score": 75.0,
      "credit_ratings": [
        {
          "agency": "Moody's",
          "rating": "Baa3",
          "outlook": "Positive",
          "date": "2026-02-14"
        },
        {
          "agency": "S&P",
          "rating": "BBB",
          "outlook": "Stable",
          "date": "2026-02-14"
        },
        {
          "agency": "Fitch",
          "rating": "BBB",
          "outlook": "Stable",
          "date": "2026-02-14"
        }
      ],
      "debt_facilities": [
        {
          "facility_type": "ABL Revolver",
          "amount_committed": 5000.0,
          "amount_drawn": 500.0,
          "interest_rate": "SOFR + 1.50%",
          "maturity_date": "2026-10-20",
          "snc_rating": "Pass",
          "drc": 0.9,
          "ltv": 0.4,
          "conviction_score": 0.88,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 57.5,
              "principal": 50.0,
              "interest": 7.5,
              "remaining_balance": 450.0
            },
            {
              "year": 2028,
              "payment_amount": 56.75,
              "principal": 50.0,
              "interest": 6.75,
              "remaining_balance": 400.0
            },
            {
              "year": 2029,
              "payment_amount": 56.0,
              "principal": 50.0,
              "interest": 6.0,
              "remaining_balance": 350.0
            },
            {
              "year": 2030,
              "payment_amount": 55.25,
              "principal": 50.0,
              "interest": 5.25,
              "remaining_balance": 300.0
            },
            {
              "year": 2031,
              "payment_amount": 54.5,
              "principal": 50.0,
              "interest": 4.5,
              "remaining_balance": 250.0
            }
          ]
        },
        {
          "facility_type": "Convertible Senior Notes",
          "amount_committed": 1800.0,
          "amount_drawn": 1800.0,
          "interest_rate": "2.00%",
          "maturity_date": "2027-05-15",
          "snc_rating": "Pass",
          "drc": 0.85,
          "ltv": 0.3,
          "conviction_score": 0.85,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 216.0,
              "principal": 180.0,
              "interest": 36.0,
              "remaining_balance": 1620.0
            },
            {
              "year": 2028,
              "payment_amount": 212.4,
              "principal": 180.0,
              "interest": 32.4,
              "remaining_balance": 1440.0
            },
            {
              "year": 2029,
              "payment_amount": 208.8,
              "principal": 180.0,
              "interest": 28.8,
              "remaining_balance": 1260.0
            },
            {
              "year": 2030,
              "payment_amount": 205.2,
              "principal": 180.0,
              "interest": 25.2,
              "remaining_balance": 1080.0
            },
            {
              "year": 2031,
              "payment_amount": 201.6,
              "principal": 180.0,
              "interest": 21.6,
              "remaining_balance": 900.0
            }
          ]
        },
        {
          "facility_type": "Auto ABS Facilities",
          "amount_committed": 3000.0,
          "amount_drawn": 2200.0,
          "interest_rate": "Variable",
          "maturity_date": "Rolling",
          "snc_rating": "Pass",
          "drc": 0.95,
          "ltv": 0.8,
          "conviction_score": 0.92,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 330.0,
              "principal": 220.0,
              "interest": 110.0,
              "remaining_balance": 1980.0
            },
            {
              "year": 2028,
              "payment_amount": 319.0,
              "principal": 220.0,
              "interest": 99.0,
              "remaining_balance": 1760.0
            },
            {
              "year": 2029,
              "payment_amount": 308.0,
              "principal": 220.0,
              "interest": 88.0,
              "remaining_balance": 1540.0
            },
            {
              "year": 2030,
              "payment_amount": 297.0,
              "principal": 220.0,
              "interest": 77.0,
              "remaining_balance": 1320.0
            },
            {
              "year": 2031,
              "payment_amount": 286.0,
              "principal": 220.0,
              "interest": 66.0,
              "remaining_balance": 1100.0
            }
          ]
        }
      ],
      "equity_data": {
        "market_cap": 850000.0,
        "share_price": 265.4,
        "volume_avg_30d": 98000000.0,
        "pe_ratio": 68.2,
        "dividend_yield": 0.0,
        "beta": 2.05
      }
    },
    "Amazoncom_Inc": {
      "borrower_name": "Amazon.com Inc.",
      "report_date": "2026-02-14T01:37:17.039190",
      "executive_summary": "The borrower Amazon.com Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($150000.0M) but elevated leverage (0.3x).  [Ref: Generic_Borrower_Profile.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower Amazon.com Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($150000.0M) but elevated leverage (0.3x).  [Ref: Generic_Borrower_Profile.pdf]",
          "citations": [
            {
              "doc_id": "Generic_Borrower_Profile.pdf",
              "chunk_id": "7d8a64d8-52c6-415a-8e2e-eff5a7f326a7",
              "page_number": 1
            }
          ],
          "author_agent": "Writer"
        },
        {
          "title": "Key Risks & Mitigants",
          "content": "Key Risks:\n",
          "citations": [],
          "author_agent": "Risk Officer"
        },
        {
          "title": "Financial Analysis",
          "content": "EBITDA: $150000.0M | Leverage: 0.3x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "financial_ratios": {
        "leverage_ratio": 0.3,
        "dscr": 60.0,
        "current_ratio": 1.5,
        "revenue": 760000.0,
        "ebitda": 150000.0,
        "net_income": 90000.0
      },
      "historical_financials": [
        {
          "total_assets": 720000.0,
          "total_liabilities": 380000.0,
          "total_equity": 340000.0,
          "revenue": 760000.0,
          "ebitda": 150000.0,
          "net_income": 90000.0,
          "interest_expense": 2500.0,
          "dscr": 60.0,
          "leverage_ratio": 0.3,
          "current_ratio": 1.5,
          "period": "FY2026"
        },
        {
          "total_assets": 662400.0,
          "total_liabilities": 349600.0,
          "total_equity": 312800.0,
          "revenue": 684000.0,
          "ebitda": 132000.0,
          "net_income": 76500.0,
          "interest_expense": 2500.0,
          "dscr": 60.0,
          "leverage_ratio": 0.3,
          "current_ratio": 1.5,
          "period": "FY2025"
        },
        {
          "total_assets": 629280.0,
          "total_liabilities": 332120.0,
          "total_equity": 297160.0,
          "revenue": 629280.0,
          "ebitda": 118800.0,
          "net_income": 67320.0,
          "interest_expense": 2500.0,
          "dscr": 60.0,
          "leverage_ratio": 0.3,
          "current_ratio": 1.5,
          "period": "FY2024"
        }
      ],
      "dcf_analysis": {
        "free_cash_flow": [
          102375.0,
          106472.43749999999,
          109674.24,
          111883.49256093746,
          113029.22224425
        ],
        "growth_rate": 0.03,
        "wacc": 0.09,
        "terminal_value": 1940334.9818596253,
        "enterprise_value": 1682033.305723774,
        "equity_value": 1302033.305723774,
        "share_price": 185.0
      },
      "risk_score": 75.0,
      "credit_ratings": [
        {
          "agency": "Moody's",
          "rating": "Ba2",
          "outlook": "Stable",
          "date": "2026-02-14"
        },
        {
          "agency": "S&P",
          "rating": "BB",
          "outlook": "Stable",
          "date": "2026-02-14"
        }
      ],
      "debt_facilities": [
        {
          "facility_type": "Revolver",
          "amount_committed": 100.0,
          "amount_drawn": 20.0,
          "interest_rate": "Prime + 1.0%",
          "maturity_date": "2026-01-01",
          "snc_rating": "Pass",
          "drc": 0.8,
          "ltv": 0.5,
          "conviction_score": 0.7,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 2.2,
              "principal": 2.0,
              "interest": 0.2,
              "remaining_balance": 18.0
            },
            {
              "year": 2028,
              "payment_amount": 2.18,
              "principal": 2.0,
              "interest": 0.18,
              "remaining_balance": 16.0
            },
            {
              "year": 2029,
              "payment_amount": 2.16,
              "principal": 2.0,
              "interest": 0.16,
              "remaining_balance": 14.0
            },
            {
              "year": 2030,
              "payment_amount": 2.14,
              "principal": 2.0,
              "interest": 0.14,
              "remaining_balance": 12.0
            },
            {
              "year": 2031,
              "payment_amount": 2.12,
              "principal": 2.0,
              "interest": 0.12,
              "remaining_balance": 10.0
            }
          ]
        },
        {
          "facility_type": "Term Loan",
          "amount_committed": 500.0,
          "amount_drawn": 500.0,
          "interest_rate": "5.50%",
          "maturity_date": "2029-01-01",
          "snc_rating": "Pass",
          "drc": 0.75,
          "ltv": 0.6,
          "conviction_score": 0.75,
          "repayment_schedule": [
            {
              "year": 2027,
              "payment_amount": 77.5,
              "principal": 50.0,
              "interest": 27.5,
              "remaining_balance": 450.0
            },
            {
              "year": 2028,
              "payment_amount": 74.75,
              "principal": 50.0,
              "interest": 24.75,
              "remaining_balance": 400.0
            },
            {
              "year": 2029,
              "payment_amount": 72.0,
              "principal": 50.0,
              "interest": 22.0,
              "remaining_balance": 350.0
            },
            {
              "year": 2030,
              "payment_amount": 69.25,
              "principal": 50.0,
              "interest": 19.25,
              "remaining_balance": 300.0
            },
            {
              "year": 2031,
              "payment_amount": 66.5,
              "principal": 50.0,
              "interest": 16.5,
              "remaining_balance": 250.0
            }
          ]
        }
      ],
      "equity_data": {
        "market_cap": 100.0,
        "share_price": 10.0,
        "volume_avg_30d": 1000.0,
        "pe_ratio": 15.0,
        "dividend_yield": 1.0,
        "beta": 1.0
      }
    }
  },
  "sovereign_data": {
    "AAPL": {
      "spread": {
        "ticker": "Apple Inc.",
        "fiscal_year": 2026,
        "metrics": {
          "Revenue": 448000,
          "EBITDA": 148000,
          "Total Debt": 90000,
          "Cash": 55000,
          "Net Debt": 35000
        },
        "growth_metrics": {
          "Revenue CAGR (3Y)": 3.2,
          "EBITDA CAGR (3Y)": 3.2,
          "Revenue YoY": 8.0,
          "EBITDA YoY": 9.6
        },
        "ratios": {
          "Leverage (Debt/EBITDA)": 0.61,
          "Interest Coverage (EBITDA/Interest)": 46.25
        },
        "valuation": {
          "dcf": {
            "enterprise_value": 1787268.21,
            "equity_value": 1752268.21,
            "share_price": 250.0,
            "wacc": 0.09,
            "growth_rate": 0.03,
            "base_fcf": 103600.0,
            "mock_shares": 7009.07283991725
          },
          "risk_model": {
            "z_score": 2.61,
            "pd_category": "Grey Zone",
            "lgd": 0.45,
            "asset_coverage": 1.32,
            "credit_rating": "A",
            "rationale": "The company exhibits a Z-Score of 2.61, placing it in the Grey Zone category. Asset coverage of 1.32x suggests a Loss Given Default (LGD) of approx 45%. Credit Rating is assessed at A. Key drivers include strong EBITDA generation relative to debt service obligations."
          },
          "forward_view": {
            "projections": [
              {
                "fiscal_year": 2027,
                "revenue": 470400.0,
                "ebitda": 155400.0
              },
              {
                "fiscal_year": 2028,
                "revenue": 493920.0,
                "ebitda": 163170.0
              },
              {
                "fiscal_year": 2029,
                "revenue": 518616.0,
                "ebitda": 171328.5
              }
            ],
            "price_targets": {
              "bull": 337.5,
              "base": 250.0,
              "bear": 162.5
            },
            "conviction_score": 73,
            "rationale": "Equity Price Target set at $250.00 (Base Case) derived from a deterministic DCF model assuming a 9.0% WACC and 3.0% terminal growth. Upside scenario (Bull) at $337.50 assumes accelerated margin expansion. Downside (Bear) at $162.50 reflects potential compression in free cash flow."
          }
        },
        "validation": {
          "identity_check": "PASS",
          "identity_delta": 0
        },
        "history": [
          {
            "fiscal_year": 2022,
            "revenue": 394328,
            "ebitda": 130541,
            "total_debt": 120069,
            "cash_equivalents": 23646,
            "interest_expense": 2931,
            "total_assets": 352755,
            "total_liabilities": 302083,
            "total_equity": 50672
          },
          {
            "fiscal_year": 2023,
            "revenue": 383285,
            "ebitda": 114301,
            "total_debt": 111088,
            "cash_equivalents": 29965,
            "interest_expense": 3933,
            "total_assets": 352583,
            "total_liabilities": 290437,
            "total_equity": 62146
          },
          {
            "fiscal_year": 2024,
            "revenue": 391000,
            "ebitda": 122000,
            "total_debt": 105000,
            "cash_equivalents": 32000,
            "interest_expense": 3800,
            "total_assets": 360000,
            "total_liabilities": 295000,
            "total_equity": 65000
          },
          {
            "fiscal_year": 2025,
            "revenue": 415000,
            "ebitda": 135000,
            "total_debt": 98000,
            "cash_equivalents": 40000,
            "interest_expense": 3500,
            "total_assets": 380000,
            "total_liabilities": 300000,
            "total_equity": 80000
          },
          {
            "fiscal_year": 2026,
            "revenue": 448000,
            "ebitda": 148000,
            "total_debt": 90000,
            "cash_equivalents": 55000,
            "interest_expense": 3200,
            "total_assets": 410000,
            "total_liabilities": 310000,
            "total_equity": 100000
          }
        ]
      },
      "memo": {
        "title": "Credit Memo: Apple Inc.",
        "date": "2026-02-14",
        "recommendation": "APPROVE",
        "executive_summary": "Borrower meets all standard covenants. Strong financial position.",
        "financial_highlights": {
          "Revenue": 448000,
          "EBITDA": 148000,
          "Total Debt": 90000,
          "Cash": 55000,
          "Net Debt": 35000
        },
        "growth_analysis": {
          "Revenue CAGR (3Y)": 3.2,
          "EBITDA CAGR (3Y)": 3.2,
          "Revenue YoY": 8.0,
          "EBITDA YoY": 9.6
        },
        "covenant_analysis": {
          "leverage_test": "PASS",
          "coverage_test": "PASS"
        }
      },
      "audit": {
        "ticker": "AAPL",
        "timestamp": "2026-02-14T01:04:14.270146",
        "quant_audit": {
          "agent_id": "agent_quant_v1",
          "action": "CALCULATE_SPREAD",
          "status": "SUCCESS",
          "details": "Leverage: 0.61x, Rev CAGR: 3.2%"
        },
        "risk_audit": {
          "agent_id": "agent_risk_officer_v1",
          "action": "POLICY_CHECK",
          "status": "SUCCESS",
          "details": "Borrower meets all standard covenants. Strong financial position."
        },
        "pipeline_status": "SUCCESS"
      },
      "report": {
        "ticker": "Apple Inc.",
        "scenarios": [
          {
            "case": "Bear",
            "probability": 0.2,
            "price_target": 162.5,
            "revenue_outlook": 403200.0,
            "description": "Recessionary environment, multiple compression, margin contraction."
          },
          {
            "case": "Base",
            "probability": 0.5,
            "price_target": 250.0,
            "revenue_outlook": 470400.0,
            "description": "Steady state growth inline with consensus estimates."
          },
          {
            "case": "Bull",
            "probability": 0.3,
            "price_target": 337.5,
            "revenue_outlook": 515200.0,
            "description": "Accelerated adoption, margin expansion, multiple re-rating."
          }
        ],
        "swot": {
          "Strengths": [
            "Strong Market Position"
          ],
          "Weaknesses": [
            "Deteriorating Solvency Metrics"
          ],
          "Opportunities": [
            "International Expansion",
            "AI Integration"
          ],
          "Threats": [
            "Regulatory Headwinds"
          ]
        },
        "cap_structure": [
          {
            "tranche": "Senior Secured Revolver",
            "amount": 45000.0,
            "priority": 1,
            "recovery_est": 100
          },
          {
            "tranche": "Senior Unsecured Notes",
            "amount": 27000.0,
            "priority": 2,
            "recovery_est": 45
          },
          {
            "tranche": "Subordinated Debt",
            "amount": 18000.0,
            "priority": 3,
            "recovery_est": 5
          }
        ],
        "citations": [
          {
            "source": "FY2023 10-K",
            "doc_id": "doc_10k_23",
            "relevance": "High"
          },
          {
            "source": "Q3 2024 Earnings Call Transcript",
            "doc_id": "doc_ec_q3_24",
            "relevance": "Medium"
          },
          {
            "source": "Moodys Credit Opinion",
            "doc_id": "doc_moodys_24",
            "relevance": "High"
          }
        ],
        "executive_summary": "Comprehensive credit analysis for Apple Inc.. Equity Price Target set at $250.00 (Base Case) derived from a deterministic DCF model assuming a 9.0% WACC and 3.0% terminal growth. Upside scenario (Bull) at $337.50 assumes accelerated margin expansion. Downside (Bear) at $162.50 reflects potential compression in free cash flow."
      }
    },
    "MSFT": {
      "spread": {
        "ticker": "Microsoft Corporation",
        "fiscal_year": 2026,
        "metrics": {
          "Revenue": 320000,
          "EBITDA": 165000,
          "Total Debt": 40000,
          "Cash": 80000,
          "Net Debt": -40000
        },
        "growth_metrics": {
          "Revenue CAGR (3Y)": 12.7,
          "EBITDA CAGR (3Y)": 14.0,
          "Revenue YoY": 14.3,
          "EBITDA YoY": 17.9
        },
        "ratios": {
          "Leverage (Debt/EBITDA)": 0.24,
          "Interest Coverage (EBITDA/Interest)": 110.0
        },
        "valuation": {
          "dcf": {
            "enterprise_value": 1992562.53,
            "equity_value": 2032562.53,
            "share_price": 250.0,
            "wacc": 0.09,
            "growth_rate": 0.03,
            "base_fcf": 115499.99999999999,
            "mock_shares": 8130.250125583421
          },
          "risk_model": {
            "z_score": 2.86,
            "pd_category": "Grey Zone",
            "lgd": 0.1,
            "asset_coverage": 2.15,
            "credit_rating": "AA",
            "rationale": "The company exhibits a Z-Score of 2.86, placing it in the Grey Zone category. Asset coverage of 2.15x suggests a Loss Given Default (LGD) of approx 10%. Credit Rating is assessed at AA. Key drivers include strong EBITDA generation relative to debt service obligations."
          },
          "forward_view": {
            "projections": [
              {
                "fiscal_year": 2027,
                "revenue": 336000.0,
                "ebitda": 173250.0
              },
              {
                "fiscal_year": 2028,
                "revenue": 352800.0,
                "ebitda": 181912.5
              },
              {
                "fiscal_year": 2029,
                "revenue": 370440.0,
                "ebitda": 191008.13
              }
            ],
            "price_targets": {
              "bull": 337.5,
              "base": 250.0,
              "bear": 162.5
            },
            "conviction_score": 78,
            "rationale": "Equity Price Target set at $250.00 (Base Case) derived from a deterministic DCF model assuming a 9.0% WACC and 3.0% terminal growth. Upside scenario (Bull) at $337.50 assumes accelerated margin expansion. Downside (Bear) at $162.50 reflects potential compression in free cash flow."
          }
        },
        "validation": {
          "identity_check": "PASS",
          "identity_delta": 0
        },
        "history": [
          {
            "fiscal_year": 2022,
            "revenue": 198270,
            "ebitda": 97843,
            "total_debt": 49751,
            "cash_equivalents": 13931,
            "interest_expense": 2063,
            "total_assets": 364840,
            "total_liabilities": 198298,
            "total_equity": 166542
          },
          {
            "fiscal_year": 2023,
            "revenue": 211915,
            "ebitda": 102384,
            "total_debt": 47204,
            "cash_equivalents": 34704,
            "interest_expense": 1968,
            "total_assets": 411976,
            "total_liabilities": 205753,
            "total_equity": 206223
          },
          {
            "fiscal_year": 2024,
            "revenue": 245000,
            "ebitda": 120000,
            "total_debt": 45000,
            "cash_equivalents": 45000,
            "interest_expense": 1800,
            "total_assets": 450000,
            "total_liabilities": 220000,
            "total_equity": 230000
          },
          {
            "fiscal_year": 2025,
            "revenue": 280000,
            "ebitda": 140000,
            "total_debt": 42000,
            "cash_equivalents": 60000,
            "interest_expense": 1600,
            "total_assets": 500000,
            "total_liabilities": 240000,
            "total_equity": 260000
          },
          {
            "fiscal_year": 2026,
            "revenue": 320000,
            "ebitda": 165000,
            "total_debt": 40000,
            "cash_equivalents": 80000,
            "interest_expense": 1500,
            "total_assets": 560000,
            "total_liabilities": 260000,
            "total_equity": 300000
          }
        ]
      },
      "memo": {
        "title": "Credit Memo: Microsoft Corporation",
        "date": "2026-02-14",
        "recommendation": "APPROVE",
        "executive_summary": "Borrower meets all standard covenants. Strong financial position.",
        "financial_highlights": {
          "Revenue": 320000,
          "EBITDA": 165000,
          "Total Debt": 40000,
          "Cash": 80000,
          "Net Debt": -40000
        },
        "growth_analysis": {
          "Revenue CAGR (3Y)": 12.7,
          "EBITDA CAGR (3Y)": 14.0,
          "Revenue YoY": 14.3,
          "EBITDA YoY": 17.9
        },
        "covenant_analysis": {
          "leverage_test": "PASS",
          "coverage_test": "PASS"
        }
      },
      "audit": {
        "ticker": "MSFT",
        "timestamp": "2026-02-14T01:04:14.272863",
        "quant_audit": {
          "agent_id": "agent_quant_v1",
          "action": "CALCULATE_SPREAD",
          "status": "SUCCESS",
          "details": "Leverage: 0.24x, Rev CAGR: 12.7%"
        },
        "risk_audit": {
          "agent_id": "agent_risk_officer_v1",
          "action": "POLICY_CHECK",
          "status": "SUCCESS",
          "details": "Borrower meets all standard covenants. Strong financial position."
        },
        "pipeline_status": "SUCCESS"
      },
      "report": {
        "ticker": "Microsoft Corporation",
        "scenarios": [
          {
            "case": "Bear",
            "probability": 0.2,
            "price_target": 162.5,
            "revenue_outlook": 288000.0,
            "description": "Recessionary environment, multiple compression, margin contraction."
          },
          {
            "case": "Base",
            "probability": 0.5,
            "price_target": 250.0,
            "revenue_outlook": 336000.0,
            "description": "Steady state growth inline with consensus estimates."
          },
          {
            "case": "Bull",
            "probability": 0.3,
            "price_target": 337.5,
            "revenue_outlook": 368000.0,
            "description": "Accelerated adoption, margin expansion, multiple re-rating."
          }
        ],
        "swot": {
          "Strengths": [
            "Strong Market Position",
            "High Asset Coverage",
            "High Revenue Growth"
          ],
          "Weaknesses": [
            "Deteriorating Solvency Metrics"
          ],
          "Opportunities": [
            "International Expansion",
            "AI Integration"
          ],
          "Threats": [
            "Regulatory Headwinds"
          ]
        },
        "cap_structure": [
          {
            "tranche": "Senior Secured Revolver",
            "amount": 20000.0,
            "priority": 1,
            "recovery_est": 100
          },
          {
            "tranche": "Senior Unsecured Notes",
            "amount": 12000.0,
            "priority": 2,
            "recovery_est": 45
          },
          {
            "tranche": "Subordinated Debt",
            "amount": 8000.0,
            "priority": 3,
            "recovery_est": 5
          }
        ],
        "citations": [
          {
            "source": "FY2023 10-K",
            "doc_id": "doc_10k_23",
            "relevance": "High"
          },
          {
            "source": "Q3 2024 Earnings Call Transcript",
            "doc_id": "doc_ec_q3_24",
            "relevance": "Medium"
          },
          {
            "source": "Moodys Credit Opinion",
            "doc_id": "doc_moodys_24",
            "relevance": "High"
          }
        ],
        "executive_summary": "Comprehensive credit analysis for Microsoft Corporation. Equity Price Target set at $250.00 (Base Case) derived from a deterministic DCF model assuming a 9.0% WACC and 3.0% terminal growth. Upside scenario (Bull) at $337.50 assumes accelerated margin expansion. Downside (Bear) at $162.50 reflects potential compression in free cash flow."
      }
    },
    "GOOGL": {
      "spread": {
        "ticker": "Alphabet Inc.",
        "fiscal_year": 2026,
        "metrics": {
          "Revenue": 395000,
          "EBITDA": 138000,
          "Total Debt": 10000,
          "Cash": 55000,
          "Net Debt": -45000
        },
        "growth_metrics": {
          "Revenue CAGR (3Y)": 8.7,
          "EBITDA CAGR (3Y)": 16.5,
          "Revenue YoY": 9.7,
          "EBITDA YoY": 15.0
        },
        "ratios": {
          "Leverage (Debt/EBITDA)": 0.07,
          "Interest Coverage (EBITDA/Interest)": 552.0
        },
        "valuation": {
          "dcf": {
            "enterprise_value": 1666506.84,
            "equity_value": 1711506.84,
            "share_price": 250.0,
            "wacc": 0.09,
            "growth_rate": 0.03,
            "base_fcf": 96600.0,
            "mock_shares": 6846.02737776068
          },
          "risk_model": {
            "z_score": 4.04,
            "pd_category": "Safe",
            "lgd": 0.1,
            "asset_coverage": 3.53,
            "credit_rating": "AAA",
            "rationale": "The company exhibits a Z-Score of 4.04, placing it in the Safe category. Asset coverage of 3.53x suggests a Loss Given Default (LGD) of approx 10%. Credit Rating is assessed at AAA. Key drivers include strong EBITDA generation relative to debt service obligations."
          },
          "forward_view": {
            "projections": [
              {
                "fiscal_year": 2027,
                "revenue": 414750.0,
                "ebitda": 144900.0
              },
              {
                "fiscal_year": 2028,
                "revenue": 435487.5,
                "ebitda": 152145.0
              },
              {
                "fiscal_year": 2029,
                "revenue": 457261.88,
                "ebitda": 159752.25
              }
            ],
            "price_targets": {
              "bull": 337.5,
              "base": 250.0,
              "bear": 162.5
            },
            "conviction_score": 80,
            "rationale": "Equity Price Target set at $250.00 (Base Case) derived from a deterministic DCF model assuming a 9.0% WACC and 3.0% terminal growth. Upside scenario (Bull) at $337.50 assumes accelerated margin expansion. Downside (Bear) at $162.50 reflects potential compression in free cash flow."
          }
        },
        "validation": {
          "identity_check": "PASS",
          "identity_delta": 0
        },
        "history": [
          {
            "fiscal_year": 2022,
            "revenue": 282836,
            "ebitda": 74842,
            "total_debt": 14701,
            "cash_equivalents": 21879,
            "interest_expense": 357,
            "total_assets": 365264,
            "total_liabilities": 109120,
            "total_equity": 256144
          },
          {
            "fiscal_year": 2023,
            "revenue": 307394,
            "ebitda": 88164,
            "total_debt": 13253,
            "cash_equivalents": 24048,
            "interest_expense": 321,
            "total_assets": 402392,
            "total_liabilities": 119048,
            "total_equity": 283344
          },
          {
            "fiscal_year": 2024,
            "revenue": 330000,
            "ebitda": 105000,
            "total_debt": 12000,
            "cash_equivalents": 30000,
            "interest_expense": 300,
            "total_assets": 440000,
            "total_liabilities": 130000,
            "total_equity": 310000
          },
          {
            "fiscal_year": 2025,
            "revenue": 360000,
            "ebitda": 120000,
            "total_debt": 11000,
            "cash_equivalents": 40000,
            "interest_expense": 280,
            "total_assets": 480000,
            "total_liabilities": 140000,
            "total_equity": 340000
          },
          {
            "fiscal_year": 2026,
            "revenue": 395000,
            "ebitda": 138000,
            "total_debt": 10000,
            "cash_equivalents": 55000,
            "interest_expense": 250,
            "total_assets": 530000,
            "total_liabilities": 150000,
            "total_equity": 380000
          }
        ]
      },
      "memo": {
        "title": "Credit Memo: Alphabet Inc.",
        "date": "2026-02-14",
        "recommendation": "APPROVE",
        "executive_summary": "Borrower meets all standard covenants. Strong financial position.",
        "financial_highlights": {
          "Revenue": 395000,
          "EBITDA": 138000,
          "Total Debt": 10000,
          "Cash": 55000,
          "Net Debt": -45000
        },
        "growth_analysis": {
          "Revenue CAGR (3Y)": 8.7,
          "EBITDA CAGR (3Y)": 16.5,
          "Revenue YoY": 9.7,
          "EBITDA YoY": 15.0
        },
        "covenant_analysis": {
          "leverage_test": "PASS",
          "coverage_test": "PASS"
        }
      },
      "audit": {
        "ticker": "GOOGL",
        "timestamp": "2026-02-14T01:04:14.275313",
        "quant_audit": {
          "agent_id": "agent_quant_v1",
          "action": "CALCULATE_SPREAD",
          "status": "SUCCESS",
          "details": "Leverage: 0.07x, Rev CAGR: 8.7%"
        },
        "risk_audit": {
          "agent_id": "agent_risk_officer_v1",
          "action": "POLICY_CHECK",
          "status": "SUCCESS",
          "details": "Borrower meets all standard covenants. Strong financial position."
        },
        "pipeline_status": "SUCCESS"
      },
      "report": {
        "ticker": "Alphabet Inc.",
        "scenarios": [
          {
            "case": "Bear",
            "probability": 0.2,
            "price_target": 162.5,
            "revenue_outlook": 355500.0,
            "description": "Recessionary environment, multiple compression, margin contraction."
          },
          {
            "case": "Base",
            "probability": 0.5,
            "price_target": 250.0,
            "revenue_outlook": 414750.0,
            "description": "Steady state growth inline with consensus estimates."
          },
          {
            "case": "Bull",
            "probability": 0.3,
            "price_target": 337.5,
            "revenue_outlook": 454250.0,
            "description": "Accelerated adoption, margin expansion, multiple re-rating."
          }
        ],
        "swot": {
          "Strengths": [
            "Strong Market Position",
            "Robust Balance Sheet (High Z-Score)",
            "High Asset Coverage"
          ],
          "Weaknesses": [],
          "Opportunities": [
            "International Expansion",
            "AI Integration"
          ],
          "Threats": [
            "Regulatory Headwinds"
          ]
        },
        "cap_structure": [
          {
            "tranche": "Senior Secured Revolver",
            "amount": 5000.0,
            "priority": 1,
            "recovery_est": 100
          },
          {
            "tranche": "Senior Unsecured Notes",
            "amount": 3000.0,
            "priority": 2,
            "recovery_est": 45
          },
          {
            "tranche": "Subordinated Debt",
            "amount": 2000.0,
            "priority": 3,
            "recovery_est": 5
          }
        ],
        "citations": [
          {
            "source": "FY2023 10-K",
            "doc_id": "doc_10k_23",
            "relevance": "High"
          },
          {
            "source": "Q3 2024 Earnings Call Transcript",
            "doc_id": "doc_ec_q3_24",
            "relevance": "Medium"
          },
          {
            "source": "Moodys Credit Opinion",
            "doc_id": "doc_moodys_24",
            "relevance": "High"
          }
        ],
        "executive_summary": "Comprehensive credit analysis for Alphabet Inc.. Equity Price Target set at $250.00 (Base Case) derived from a deterministic DCF model assuming a 9.0% WACC and 3.0% terminal growth. Upside scenario (Bull) at $337.50 assumes accelerated margin expansion. Downside (Bear) at $162.50 reflects potential compression in free cash flow."
      }
    },
    "AMZN": {
      "spread": {
        "ticker": "Amazon.com, Inc.",
        "fiscal_year": 2026,
        "metrics": {
          "Revenue": 760000,
          "EBITDA": 150000,
          "Total Debt": 45000,
          "Cash": 120000,
          "Net Debt": -75000
        },
        "growth_metrics": {
          "Revenue CAGR (3Y)": 10.3,
          "EBITDA CAGR (3Y)": 29.0,
          "Revenue YoY": 11.8,
          "EBITDA YoY": 20.0
        },
        "ratios": {
          "Leverage (Debt/EBITDA)": 0.3,
          "Interest Coverage (EBITDA/Interest)": 60.0
        },
        "valuation": {
          "dcf": {
            "enterprise_value": 1811420.48,
            "equity_value": 1886420.48,
            "share_price": 250.0,
            "wacc": 0.09,
            "growth_rate": 0.03,
            "base_fcf": 105000.0,
            "mock_shares": 7545.681932348564
          },
          "risk_model": {
            "z_score": 2.86,
            "pd_category": "Grey Zone",
            "lgd": 0.45,
            "asset_coverage": 1.89,
            "credit_rating": "AA",
            "rationale": "The company exhibits a Z-Score of 2.86, placing it in the Grey Zone category. Asset coverage of 1.89x suggests a Loss Given Default (LGD) of approx 45%. Credit Rating is assessed at AA. Key drivers include strong EBITDA generation relative to debt service obligations."
          },
          "forward_view": {
            "projections": [
              {
                "fiscal_year": 2027,
                "revenue": 798000.0,
                "ebitda": 157500.0
              },
              {
                "fiscal_year": 2028,
                "revenue": 837900.0,
                "ebitda": 165375.0
              },
              {
                "fiscal_year": 2029,
                "revenue": 879795.0,
                "ebitda": 173643.75
              }
            ],
            "price_targets": {
              "bull": 337.5,
              "base": 250.0,
              "bear": 162.5
            },
            "conviction_score": 78,
            "rationale": "Equity Price Target set at $250.00 (Base Case) derived from a deterministic DCF model assuming a 9.0% WACC and 3.0% terminal growth. Upside scenario (Bull) at $337.50 assumes accelerated margin expansion. Downside (Bear) at $162.50 reflects potential compression in free cash flow."
          }
        },
        "validation": {
          "identity_check": "PASS",
          "identity_delta": 0
        },
        "history": [
          {
            "fiscal_year": 2022,
            "revenue": 513983,
            "ebitda": 54169,
            "total_debt": 67150,
            "cash_equivalents": 53888,
            "interest_expense": 2367,
            "total_assets": 462675,
            "total_liabilities": 316632,
            "total_equity": 146043
          },
          {
            "fiscal_year": 2023,
            "revenue": 574785,
            "ebitda": 85515,
            "total_debt": 58316,
            "cash_equivalents": 73387,
            "interest_expense": 3178,
            "total_assets": 527854,
            "total_liabilities": 326084,
            "total_equity": 201770
          },
          {
            "fiscal_year": 2024,
            "revenue": 620000,
            "ebitda": 105000,
            "total_debt": 55000,
            "cash_equivalents": 85000,
            "interest_expense": 3000,
            "total_assets": 580000,
            "total_liabilities": 340000,
            "total_equity": 240000
          },
          {
            "fiscal_year": 2025,
            "revenue": 680000,
            "ebitda": 125000,
            "total_debt": 50000,
            "cash_equivalents": 100000,
            "interest_expense": 2800,
            "total_assets": 640000,
            "total_liabilities": 360000,
            "total_equity": 280000
          },
          {
            "fiscal_year": 2026,
            "revenue": 760000,
            "ebitda": 150000,
            "total_debt": 45000,
            "cash_equivalents": 120000,
            "interest_expense": 2500,
            "total_assets": 720000,
            "total_liabilities": 380000,
            "total_equity": 340000
          }
        ]
      },
      "memo": {
        "title": "Credit Memo: Amazon.com, Inc.",
        "date": "2026-02-14",
        "recommendation": "APPROVE",
        "executive_summary": "Borrower meets all standard covenants. Strong financial position.",
        "financial_highlights": {
          "Revenue": 760000,
          "EBITDA": 150000,
          "Total Debt": 45000,
          "Cash": 120000,
          "Net Debt": -75000
        },
        "growth_analysis": {
          "Revenue CAGR (3Y)": 10.3,
          "EBITDA CAGR (3Y)": 29.0,
          "Revenue YoY": 11.8,
          "EBITDA YoY": 20.0
        },
        "covenant_analysis": {
          "leverage_test": "PASS",
          "coverage_test": "PASS"
        }
      },
      "audit": {
        "ticker": "AMZN",
        "timestamp": "2026-02-14T01:04:14.277740",
        "quant_audit": {
          "agent_id": "agent_quant_v1",
          "action": "CALCULATE_SPREAD",
          "status": "SUCCESS",
          "details": "Leverage: 0.30x, Rev CAGR: 10.3%"
        },
        "risk_audit": {
          "agent_id": "agent_risk_officer_v1",
          "action": "POLICY_CHECK",
          "status": "SUCCESS",
          "details": "Borrower meets all standard covenants. Strong financial position."
        },
        "pipeline_status": "SUCCESS"
      },
      "report": {
        "ticker": "Amazon.com, Inc.",
        "scenarios": [
          {
            "case": "Bear",
            "probability": 0.2,
            "price_target": 162.5,
            "revenue_outlook": 684000.0,
            "description": "Recessionary environment, multiple compression, margin contraction."
          },
          {
            "case": "Base",
            "probability": 0.5,
            "price_target": 250.0,
            "revenue_outlook": 798000.0,
            "description": "Steady state growth inline with consensus estimates."
          },
          {
            "case": "Bull",
            "probability": 0.3,
            "price_target": 337.5,
            "revenue_outlook": 874000.0,
            "description": "Accelerated adoption, margin expansion, multiple re-rating."
          }
        ],
        "swot": {
          "Strengths": [
            "Strong Market Position",
            "High Revenue Growth"
          ],
          "Weaknesses": [
            "Deteriorating Solvency Metrics"
          ],
          "Opportunities": [
            "International Expansion",
            "AI Integration"
          ],
          "Threats": [
            "Regulatory Headwinds"
          ]
        },
        "cap_structure": [
          {
            "tranche": "Senior Secured Revolver",
            "amount": 22500.0,
            "priority": 1,
            "recovery_est": 100
          },
          {
            "tranche": "Senior Unsecured Notes",
            "amount": 13500.0,
            "priority": 2,
            "recovery_est": 45
          },
          {
            "tranche": "Subordinated Debt",
            "amount": 9000.0,
            "priority": 3,
            "recovery_est": 5
          }
        ],
        "citations": [
          {
            "source": "FY2023 10-K",
            "doc_id": "doc_10k_23",
            "relevance": "High"
          },
          {
            "source": "Q3 2024 Earnings Call Transcript",
            "doc_id": "doc_ec_q3_24",
            "relevance": "Medium"
          },
          {
            "source": "Moodys Credit Opinion",
            "doc_id": "doc_moodys_24",
            "relevance": "High"
          }
        ],
        "executive_summary": "Comprehensive credit analysis for Amazon.com, Inc.. Equity Price Target set at $250.00 (Base Case) derived from a deterministic DCF model assuming a 9.0% WACC and 3.0% terminal growth. Upside scenario (Bull) at $337.50 assumes accelerated margin expansion. Downside (Bear) at $162.50 reflects potential compression in free cash flow."
      }
    },
    "NVDA": {
      "spread": {
        "ticker": "NVIDIA Corporation",
        "fiscal_year": 2026,
        "metrics": {
          "Revenue": 140000,
          "EBITDA": 92000,
          "Total Debt": 7500,
          "Cash": 65000,
          "Net Debt": -57500
        },
        "growth_metrics": {
          "Revenue CAGR (3Y)": 51.0,
          "EBITDA CAGR (3Y)": 69.2,
          "Revenue YoY": 47.4,
          "EBITDA YoY": 58.6
        },
        "ratios": {
          "Leverage (Debt/EBITDA)": 0.08,
          "Interest Coverage (EBITDA/Interest)": 418.18
        },
        "valuation": {
          "dcf": {
            "enterprise_value": 1111004.56,
            "equity_value": 1168504.56,
            "share_price": 250.0,
            "wacc": 0.09,
            "growth_rate": 0.03,
            "base_fcf": 64399.99999999999,
            "mock_shares": 4674.018251840452
          },
          "risk_model": {
            "z_score": 6.11,
            "pd_category": "Safe",
            "lgd": 0.1,
            "asset_coverage": 4.33,
            "credit_rating": "AAA",
            "rationale": "The company exhibits a Z-Score of 6.11, placing it in the Safe category. Asset coverage of 4.33x suggests a Loss Given Default (LGD) of approx 10%. Credit Rating is assessed at AAA. Key drivers include strong EBITDA generation relative to debt service obligations."
          },
          "forward_view": {
            "projections": [
              {
                "fiscal_year": 2027,
                "revenue": 147000.0,
                "ebitda": 96600.0
              },
              {
                "fiscal_year": 2028,
                "revenue": 154350.0,
                "ebitda": 101430.0
              },
              {
                "fiscal_year": 2029,
                "revenue": 162067.5,
                "ebitda": 106501.5
              }
            ],
            "price_targets": {
              "bull": 337.5,
              "base": 250.0,
              "bear": 162.5
            },
            "conviction_score": 80,
            "rationale": "Equity Price Target set at $250.00 (Base Case) derived from a deterministic DCF model assuming a 9.0% WACC and 3.0% terminal growth. Upside scenario (Bull) at $337.50 assumes accelerated margin expansion. Downside (Bear) at $162.50 reflects potential compression in free cash flow."
          }
        },
        "validation": {
          "identity_check": "PASS",
          "identity_delta": 0
        },
        "history": [
          {
            "fiscal_year": 2022,
            "revenue": 26914,
            "ebitda": 11216,
            "total_debt": 10946,
            "cash_equivalents": 1991,
            "interest_expense": 236,
            "total_assets": 44187,
            "total_liabilities": 17575,
            "total_equity": 26612
          },
          {
            "fiscal_year": 2023,
            "revenue": 26974,
            "ebitda": 5600,
            "total_debt": 11130,
            "cash_equivalents": 3389,
            "interest_expense": 272,
            "total_assets": 41182,
            "total_liabilities": 19081,
            "total_equity": 22101
          },
          {
            "fiscal_year": 2024,
            "revenue": 60922,
            "ebitda": 34480,
            "total_debt": 8461,
            "cash_equivalents": 25984,
            "interest_expense": 257,
            "total_assets": 65728,
            "total_liabilities": 22750,
            "total_equity": 42978
          },
          {
            "fiscal_year": 2025,
            "revenue": 95000,
            "ebitda": 58000,
            "total_debt": 8000,
            "cash_equivalents": 40000,
            "interest_expense": 240,
            "total_assets": 90000,
            "total_liabilities": 25000,
            "total_equity": 65000
          },
          {
            "fiscal_year": 2026,
            "revenue": 140000,
            "ebitda": 92000,
            "total_debt": 7500,
            "cash_equivalents": 65000,
            "interest_expense": 220,
            "total_assets": 130000,
            "total_liabilities": 30000,
            "total_equity": 100000
          }
        ]
      },
      "memo": {
        "title": "Credit Memo: NVIDIA Corporation",
        "date": "2026-02-14",
        "recommendation": "APPROVE",
        "executive_summary": "Borrower meets all standard covenants. Strong financial position.",
        "financial_highlights": {
          "Revenue": 140000,
          "EBITDA": 92000,
          "Total Debt": 7500,
          "Cash": 65000,
          "Net Debt": -57500
        },
        "growth_analysis": {
          "Revenue CAGR (3Y)": 51.0,
          "EBITDA CAGR (3Y)": 69.2,
          "Revenue YoY": 47.4,
          "EBITDA YoY": 58.6
        },
        "covenant_analysis": {
          "leverage_test": "PASS",
          "coverage_test": "PASS"
        }
      },
      "audit": {
        "ticker": "NVDA",
        "timestamp": "2026-02-14T01:04:14.280264",
        "quant_audit": {
          "agent_id": "agent_quant_v1",
          "action": "CALCULATE_SPREAD",
          "status": "SUCCESS",
          "details": "Leverage: 0.08x, Rev CAGR: 51.0%"
        },
        "risk_audit": {
          "agent_id": "agent_risk_officer_v1",
          "action": "POLICY_CHECK",
          "status": "SUCCESS",
          "details": "Borrower meets all standard covenants. Strong financial position."
        },
        "pipeline_status": "SUCCESS"
      },
      "report": {
        "ticker": "NVIDIA Corporation",
        "scenarios": [
          {
            "case": "Bear",
            "probability": 0.2,
            "price_target": 162.5,
            "revenue_outlook": 126000.0,
            "description": "Recessionary environment, multiple compression, margin contraction."
          },
          {
            "case": "Base",
            "probability": 0.5,
            "price_target": 250.0,
            "revenue_outlook": 147000.0,
            "description": "Steady state growth inline with consensus estimates."
          },
          {
            "case": "Bull",
            "probability": 0.3,
            "price_target": 337.5,
            "revenue_outlook": 161000.0,
            "description": "Accelerated adoption, margin expansion, multiple re-rating."
          }
        ],
        "swot": {
          "Strengths": [
            "Strong Market Position",
            "Robust Balance Sheet (High Z-Score)",
            "High Asset Coverage",
            "High Revenue Growth"
          ],
          "Weaknesses": [],
          "Opportunities": [
            "International Expansion",
            "AI Integration"
          ],
          "Threats": [
            "Regulatory Headwinds"
          ]
        },
        "cap_structure": [
          {
            "tranche": "Senior Secured Revolver",
            "amount": 3750.0,
            "priority": 1,
            "recovery_est": 100
          },
          {
            "tranche": "Senior Unsecured Notes",
            "amount": 2250.0,
            "priority": 2,
            "recovery_est": 45
          },
          {
            "tranche": "Subordinated Debt",
            "amount": 1500.0,
            "priority": 3,
            "recovery_est": 5
          }
        ],
        "citations": [
          {
            "source": "FY2023 10-K",
            "doc_id": "doc_10k_23",
            "relevance": "High"
          },
          {
            "source": "Q3 2024 Earnings Call Transcript",
            "doc_id": "doc_ec_q3_24",
            "relevance": "Medium"
          },
          {
            "source": "Moodys Credit Opinion",
            "doc_id": "doc_moodys_24",
            "relevance": "High"
          }
        ],
        "executive_summary": "Comprehensive credit analysis for NVIDIA Corporation. Equity Price Target set at $250.00 (Base Case) derived from a deterministic DCF model assuming a 9.0% WACC and 3.0% terminal growth. Upside scenario (Bull) at $337.50 assumes accelerated margin expansion. Downside (Bear) at $162.50 reflects potential compression in free cash flow."
      }
    },
    "TSLA": {
      "spread": {
        "ticker": "Tesla, Inc.",
        "fiscal_year": 2026,
        "metrics": {
          "Revenue": 160000,
          "EBITDA": 32000,
          "Total Debt": 7000,
          "Cash": 45000,
          "Net Debt": -38000
        },
        "growth_metrics": {
          "Revenue CAGR (3Y)": 18.4,
          "EBITDA CAGR (3Y)": 16.0,
          "Revenue YoY": 18.5,
          "EBITDA YoY": 33.3
        },
        "ratios": {
          "Leverage (Debt/EBITDA)": 0.22,
          "Interest Coverage (EBITDA/Interest)": 106.67
        },
        "valuation": {
          "dcf": {
            "enterprise_value": 386436.37,
            "equity_value": 424436.37,
            "share_price": 250.0,
            "wacc": 0.09,
            "growth_rate": 0.03,
            "base_fcf": 22400.0,
            "mock_shares": 1697.7454789010267
          },
          "risk_model": {
            "z_score": 3.23,
            "pd_category": "Safe",
            "lgd": 0.1,
            "asset_coverage": 2.57,
            "credit_rating": "AAA",
            "rationale": "The company exhibits a Z-Score of 3.23, placing it in the Safe category. Asset coverage of 2.57x suggests a Loss Given Default (LGD) of approx 10%. Credit Rating is assessed at AAA. Key drivers include strong EBITDA generation relative to debt service obligations."
          },
          "forward_view": {
            "projections": [
              {
                "fiscal_year": 2027,
                "revenue": 168000.0,
                "ebitda": 33600.0
              },
              {
                "fiscal_year": 2028,
                "revenue": 176400.0,
                "ebitda": 35280.0
              },
              {
                "fiscal_year": 2029,
                "revenue": 185220.0,
                "ebitda": 37044.0
              }
            ],
            "price_targets": {
              "bull": 337.5,
              "base": 250.0,
              "bear": 162.5
            },
            "conviction_score": 80,
            "rationale": "Equity Price Target set at $250.00 (Base Case) derived from a deterministic DCF model assuming a 9.0% WACC and 3.0% terminal growth. Upside scenario (Bull) at $337.50 assumes accelerated margin expansion. Downside (Bear) at $162.50 reflects potential compression in free cash flow."
          }
        },
        "validation": {
          "identity_check": "PASS",
          "identity_delta": 0
        },
        "history": [
          {
            "fiscal_year": 2022,
            "revenue": 81462,
            "ebitda": 17660,
            "total_debt": 3099,
            "cash_equivalents": 22185,
            "interest_expense": 191,
            "total_assets": 82338,
            "total_liabilities": 36440,
            "total_equity": 44704
          },
          {
            "fiscal_year": 2023,
            "revenue": 96773,
            "ebitda": 14997,
            "total_debt": 4350,
            "cash_equivalents": 29072,
            "interest_expense": 156,
            "total_assets": 106618,
            "total_liabilities": 43009,
            "total_equity": 62634
          },
          {
            "fiscal_year": 2024,
            "revenue": 110000,
            "ebitda": 18000,
            "total_debt": 5000,
            "cash_equivalents": 32000,
            "interest_expense": 200,
            "total_assets": 125000,
            "total_liabilities": 50000,
            "total_equity": 75000
          },
          {
            "fiscal_year": 2025,
            "revenue": 135000,
            "ebitda": 24000,
            "total_debt": 6000,
            "cash_equivalents": 38000,
            "interest_expense": 250,
            "total_assets": 150000,
            "total_liabilities": 60000,
            "total_equity": 90000
          },
          {
            "fiscal_year": 2026,
            "revenue": 160000,
            "ebitda": 32000,
            "total_debt": 7000,
            "cash_equivalents": 45000,
            "interest_expense": 300,
            "total_assets": 180000,
            "total_liabilities": 70000,
            "total_equity": 110000
          }
        ]
      },
      "memo": {
        "title": "Credit Memo: Tesla, Inc.",
        "date": "2026-02-14",
        "recommendation": "APPROVE",
        "executive_summary": "Borrower meets all standard covenants. Strong financial position.",
        "financial_highlights": {
          "Revenue": 160000,
          "EBITDA": 32000,
          "Total Debt": 7000,
          "Cash": 45000,
          "Net Debt": -38000
        },
        "growth_analysis": {
          "Revenue CAGR (3Y)": 18.4,
          "EBITDA CAGR (3Y)": 16.0,
          "Revenue YoY": 18.5,
          "EBITDA YoY": 33.3
        },
        "covenant_analysis": {
          "leverage_test": "PASS",
          "coverage_test": "PASS"
        }
      },
      "audit": {
        "ticker": "TSLA",
        "timestamp": "2026-02-14T01:04:14.282685",
        "quant_audit": {
          "agent_id": "agent_quant_v1",
          "action": "CALCULATE_SPREAD",
          "status": "SUCCESS",
          "details": "Leverage: 0.22x, Rev CAGR: 18.4%"
        },
        "risk_audit": {
          "agent_id": "agent_risk_officer_v1",
          "action": "POLICY_CHECK",
          "status": "SUCCESS",
          "details": "Borrower meets all standard covenants. Strong financial position."
        },
        "pipeline_status": "SUCCESS"
      },
      "report": {
        "ticker": "Tesla, Inc.",
        "scenarios": [
          {
            "case": "Bear",
            "probability": 0.2,
            "price_target": 162.5,
            "revenue_outlook": 144000.0,
            "description": "Recessionary environment, multiple compression, margin contraction."
          },
          {
            "case": "Base",
            "probability": 0.5,
            "price_target": 250.0,
            "revenue_outlook": 168000.0,
            "description": "Steady state growth inline with consensus estimates."
          },
          {
            "case": "Bull",
            "probability": 0.3,
            "price_target": 337.5,
            "revenue_outlook": 184000.0,
            "description": "Accelerated adoption, margin expansion, multiple re-rating."
          }
        ],
        "swot": {
          "Strengths": [
            "Strong Market Position",
            "Robust Balance Sheet (High Z-Score)",
            "High Asset Coverage",
            "High Revenue Growth"
          ],
          "Weaknesses": [],
          "Opportunities": [
            "International Expansion",
            "AI Integration"
          ],
          "Threats": [
            "Regulatory Headwinds"
          ]
        },
        "cap_structure": [
          {
            "tranche": "Senior Secured Revolver",
            "amount": 3500.0,
            "priority": 1,
            "recovery_est": 100
          },
          {
            "tranche": "Senior Unsecured Notes",
            "amount": 2100.0,
            "priority": 2,
            "recovery_est": 45
          },
          {
            "tranche": "Subordinated Debt",
            "amount": 1400.0,
            "priority": 3,
            "recovery_est": 5
          }
        ],
        "citations": [
          {
            "source": "FY2023 10-K",
            "doc_id": "doc_10k_23",
            "relevance": "High"
          },
          {
            "source": "Q3 2024 Earnings Call Transcript",
            "doc_id": "doc_ec_q3_24",
            "relevance": "Medium"
          },
          {
            "source": "Moodys Credit Opinion",
            "doc_id": "doc_moodys_24",
            "relevance": "High"
          }
        ],
        "executive_summary": "Comprehensive credit analysis for Tesla, Inc.. Equity Price Target set at $250.00 (Base Case) derived from a deterministic DCF model assuming a 9.0% WACC and 3.0% terminal growth. Upside scenario (Bull) at $337.50 assumes accelerated margin expansion. Downside (Bear) at $162.50 reflects potential compression in free cash flow."
      }
    },
    "META": {
      "spread": {
        "ticker": "Meta Platforms, Inc.",
        "fiscal_year": 2026,
        "metrics": {
          "Revenue": 210000,
          "EBITDA": 110000,
          "Total Debt": 30000,
          "Cash": 85000,
          "Net Debt": -55000
        },
        "growth_metrics": {
          "Revenue CAGR (3Y)": 15.8,
          "EBITDA CAGR (3Y)": 28.5,
          "Revenue YoY": 16.7,
          "EBITDA YoY": 22.2
        },
        "ratios": {
          "Leverage (Debt/EBITDA)": 0.27,
          "Interest Coverage (EBITDA/Interest)": 366.67
        },
        "valuation": {
          "dcf": {
            "enterprise_value": 1328375.02,
            "equity_value": 1383375.02,
            "share_price": 250.0,
            "wacc": 0.09,
            "growth_rate": 0.03,
            "base_fcf": 77000.0,
            "mock_shares": 5533.5000837222815
          },
          "risk_model": {
            "z_score": 3.89,
            "pd_category": "Safe",
            "lgd": 0.1,
            "asset_coverage": 3.33,
            "credit_rating": "AAA",
            "rationale": "The company exhibits a Z-Score of 3.89, placing it in the Safe category. Asset coverage of 3.33x suggests a Loss Given Default (LGD) of approx 10%. Credit Rating is assessed at AAA. Key drivers include strong EBITDA generation relative to debt service obligations."
          },
          "forward_view": {
            "projections": [
              {
                "fiscal_year": 2027,
                "revenue": 220500.0,
                "ebitda": 115500.0
              },
              {
                "fiscal_year": 2028,
                "revenue": 231525.0,
                "ebitda": 121275.0
              },
              {
                "fiscal_year": 2029,
                "revenue": 243101.25,
                "ebitda": 127338.75
              }
            ],
            "price_targets": {
              "bull": 337.5,
              "base": 250.0,
              "bear": 162.5
            },
            "conviction_score": 80,
            "rationale": "Equity Price Target set at $250.00 (Base Case) derived from a deterministic DCF model assuming a 9.0% WACC and 3.0% terminal growth. Upside scenario (Bull) at $337.50 assumes accelerated margin expansion. Downside (Bear) at $162.50 reflects potential compression in free cash flow."
          }
        },
        "validation": {
          "identity_check": "PASS",
          "identity_delta": 0
        },
        "history": [
          {
            "fiscal_year": 2022,
            "revenue": 116609,
            "ebitda": 40380,
            "total_debt": 26402,
            "cash_equivalents": 14681,
            "interest_expense": 109,
            "total_assets": 185727,
            "total_liabilities": 60014,
            "total_equity": 125713
          },
          {
            "fiscal_year": 2023,
            "revenue": 134902,
            "ebitda": 62310,
            "total_debt": 37043,
            "cash_equivalents": 41862,
            "interest_expense": 371,
            "total_assets": 229623,
            "total_liabilities": 76016,
            "total_equity": 153607
          },
          {
            "fiscal_year": 2024,
            "revenue": 155000,
            "ebitda": 75000,
            "total_debt": 35000,
            "cash_equivalents": 50000,
            "interest_expense": 350,
            "total_assets": 260000,
            "total_liabilities": 85000,
            "total_equity": 175000
          },
          {
            "fiscal_year": 2025,
            "revenue": 180000,
            "ebitda": 90000,
            "total_debt": 32000,
            "cash_equivalents": 65000,
            "interest_expense": 320,
            "total_assets": 300000,
            "total_liabilities": 95000,
            "total_equity": 205000
          },
          {
            "fiscal_year": 2026,
            "revenue": 210000,
            "ebitda": 110000,
            "total_debt": 30000,
            "cash_equivalents": 85000,
            "interest_expense": 300,
            "total_assets": 350000,
            "total_liabilities": 105000,
            "total_equity": 245000
          }
        ]
      },
      "memo": {
        "title": "Credit Memo: Meta Platforms, Inc.",
        "date": "2026-02-14",
        "recommendation": "APPROVE",
        "executive_summary": "Borrower meets all standard covenants. Strong financial position.",
        "financial_highlights": {
          "Revenue": 210000,
          "EBITDA": 110000,
          "Total Debt": 30000,
          "Cash": 85000,
          "Net Debt": -55000
        },
        "growth_analysis": {
          "Revenue CAGR (3Y)": 15.8,
          "EBITDA CAGR (3Y)": 28.5,
          "Revenue YoY": 16.7,
          "EBITDA YoY": 22.2
        },
        "covenant_analysis": {
          "leverage_test": "PASS",
          "coverage_test": "PASS"
        }
      },
      "audit": {
        "ticker": "META",
        "timestamp": "2026-02-14T01:04:14.285093",
        "quant_audit": {
          "agent_id": "agent_quant_v1",
          "action": "CALCULATE_SPREAD",
          "status": "SUCCESS",
          "details": "Leverage: 0.27x, Rev CAGR: 15.8%"
        },
        "risk_audit": {
          "agent_id": "agent_risk_officer_v1",
          "action": "POLICY_CHECK",
          "status": "SUCCESS",
          "details": "Borrower meets all standard covenants. Strong financial position."
        },
        "pipeline_status": "SUCCESS"
      },
      "report": {
        "ticker": "Meta Platforms, Inc.",
        "scenarios": [
          {
            "case": "Bear",
            "probability": 0.2,
            "price_target": 162.5,
            "revenue_outlook": 189000.0,
            "description": "Recessionary environment, multiple compression, margin contraction."
          },
          {
            "case": "Base",
            "probability": 0.5,
            "price_target": 250.0,
            "revenue_outlook": 220500.0,
            "description": "Steady state growth inline with consensus estimates."
          },
          {
            "case": "Bull",
            "probability": 0.3,
            "price_target": 337.5,
            "revenue_outlook": 241500.0,
            "description": "Accelerated adoption, margin expansion, multiple re-rating."
          }
        ],
        "swot": {
          "Strengths": [
            "Strong Market Position",
            "Robust Balance Sheet (High Z-Score)",
            "High Asset Coverage",
            "High Revenue Growth"
          ],
          "Weaknesses": [],
          "Opportunities": [
            "International Expansion",
            "AI Integration"
          ],
          "Threats": [
            "Regulatory Headwinds"
          ]
        },
        "cap_structure": [
          {
            "tranche": "Senior Secured Revolver",
            "amount": 15000.0,
            "priority": 1,
            "recovery_est": 100
          },
          {
            "tranche": "Senior Unsecured Notes",
            "amount": 9000.0,
            "priority": 2,
            "recovery_est": 45
          },
          {
            "tranche": "Subordinated Debt",
            "amount": 6000.0,
            "priority": 3,
            "recovery_est": 5
          }
        ],
        "citations": [
          {
            "source": "FY2023 10-K",
            "doc_id": "doc_10k_23",
            "relevance": "High"
          },
          {
            "source": "Q3 2024 Earnings Call Transcript",
            "doc_id": "doc_ec_q3_24",
            "relevance": "Medium"
          },
          {
            "source": "Moodys Credit Opinion",
            "doc_id": "doc_moodys_24",
            "relevance": "High"
          }
        ],
        "executive_summary": "Comprehensive credit analysis for Meta Platforms, Inc.. Equity Price Target set at $250.00 (Base Case) derived from a deterministic DCF model assuming a 9.0% WACC and 3.0% terminal growth. Upside scenario (Bull) at $337.50 assumes accelerated margin expansion. Downside (Bear) at $162.50 reflects potential compression in free cash flow."
      }
    }
  },
  "market_mayhem": {
    "v23_knowledge_graph": {
      "meta": {
        "target": "MARKET_MAYHEM_MACRO_STRATEGY",
        "generated_at": "2025-12-14T18:40:00Z",
        "model_version": "Adam-v23.5-Apex"
      },
      "nodes": {
        "macro_ecosystem": {
          "regime_classification": {
            "status": "Great Divergence",
            "inflation_vector": "Sticky Services / Stagflationary Tails",
            "geopolitical_state": "Endogenous Risk Variable"
          },
          "consumer_health": {
            "excess_savings": "Depleted (Bottom 80%)",
            "delinquency_metrics": {
              "aggregate_rate": "2.98%",
              "sub_aggregate_breakout": {
                "majority_black_tracts": "4.8%",
                "majority_hispanic_tracts": "4.5%",
                "majority_white_tracts": "2.4%"
              },
              "divergence_verdict": "K-Shaped / High Stress in Lower Income Deciles"
            }
          }
        },
        "equity_analysis": {
          "valuation_engine": {
            "sp500_metrics": {
              "forward_pe": 23.0,
              "verdict": "Priced for Perfection"
            },
            "conviction_list": [
              {
                "ticker": "VOLT",
                "name": "Volta Motors",
                "target": 185.0,
                "thesis": "Solid-State Battery / Spinoff"
              },
              {
                "ticker": "CRML",
                "name": "Critical Metals Corp",
                "target": 45.0,
                "thesis": "Tanbreez / Non-Chinese REE Supply Chain"
              },
              {
                "ticker": "BK",
                "name": "BNY Mellon",
                "target": 95.0,
                "thesis": "Operational Alpha via Eliza/Gemini AI"
              }
            ]
          }
        },
        "credit_analysis": {
          "snc_rating_model": {
            "sector_focus": "PE-Backed Roll-ups (Healthcare/Software)",
            "market_observation": "EBITDA Mirage",
            "dscr_trend": "< 1.0x (Zombie Status)",
            "primary_risk": "Revolver dependency for interest payments"
          }
        },
        "simulation_engine": {
          "quantum_scenarios": [
            {
              "scenario_name": "Operation Midnight Hammer",
              "status": "Active / Realized",
              "impact": "Brent Crude Floor $90+"
            },
            {
              "scenario_name": "Cyber Paralysis",
              "probability": "Medium",
              "impact_severity": "Critical",
              "hedge": "AWAV (AlphaWave)"
            }
          ]
        },
        "strategic_synthesis": {
          "allocation_mandate": "70/30",
          "fortress_allocation": {
            "weight": "70%",
            "components": [
              "Private Credit (25%)",
              "Real Assets (30%)",
              "Strategic Cash (15%)"
            ]
          },
          "hunt_allocation": {
            "weight": "30%",
            "components": [
              "Deep Tech (12%)",
              "Distressed Credit (10%)",
              "Speculative (8%)"
            ]
          },
          "final_verdict": {
            "recommendation": "Defensive Growth",
            "conviction_level": 9,
            "rationale_summary": "Long thesis predicated on margin expansion in AI-adopters and hard asset inflation, hedged against consumer credit breakage."
          }
        }
      }
    }
  }
};
