window.MOCK_DATA = {
  "stats": {
    "version": "26.0 (Sovereign)",
    "generated_at": "2026-02-14T02:05:07.788738",
    "status": "ONLINE"
  },
  "credit_library": [
    {
      "id": "Apple_Inc",
      "borrower_name": "Apple Inc.",
      "ticker": "AAPL",
      "sector": "Technology",
      "report_date": "2026-02-14T02:03:52.670838",
      "risk_score": 80.0,
      "file": "credit_memo_Apple_Inc.json",
      "summary": "The borrower Apple Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($155000.0M) but elevated leverage (0.9x).  [Ref: A..."
    },
    {
      "id": "Microsoft_Corp",
      "borrower_name": "Microsoft Corp",
      "ticker": "MSFT",
      "sector": "Technology",
      "report_date": "2026-02-14T02:03:52.672795",
      "risk_score": 60.0,
      "file": "credit_memo_Microsoft_Corp.json",
      "summary": "The borrower Microsoft Corp presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: T..."
    },
    {
      "id": "NVIDIA_Corp",
      "borrower_name": "NVIDIA Corp",
      "ticker": "NVDA",
      "sector": "Technology",
      "report_date": "2026-02-14T02:03:52.674247",
      "risk_score": 60.0,
      "file": "credit_memo_NVIDIA_Corp.json",
      "summary": "The borrower NVIDIA Corp presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: Tech..."
    },
    {
      "id": "Alphabet_Inc",
      "borrower_name": "Alphabet Inc.",
      "ticker": "GOOGL",
      "sector": "Technology",
      "report_date": "2026-02-14T02:03:52.675494",
      "risk_score": 80.0,
      "file": "credit_memo_Alphabet_Inc.json",
      "summary": "The borrower Alphabet Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($138000.0M) but elevated leverage (0.0x).  [Ref..."
    },
    {
      "id": "Amazoncom_Inc",
      "borrower_name": "Amazon.com Inc.",
      "ticker": "AMZN",
      "sector": "Consumer",
      "report_date": "2026-02-14T02:03:52.676637",
      "risk_score": 60.0,
      "file": "credit_memo_Amazoncom_Inc.json",
      "summary": "The borrower Amazon.com Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: ..."
    },
    {
      "id": "Tesla_Inc",
      "borrower_name": "Tesla Inc.",
      "ticker": "TSLA",
      "sector": "Consumer",
      "report_date": "2026-02-14T02:03:52.678370",
      "risk_score": 70.0,
      "file": "credit_memo_Tesla_Inc.json",
      "summary": "The borrower Tesla Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($12500.0M) but elevated leverage (3.7x).  [Ref: TS..."
    },
    {
      "id": "Meta_Platforms",
      "borrower_name": "Meta Platforms",
      "ticker": "META",
      "sector": "Technology",
      "report_date": "2026-02-14T02:03:52.679686",
      "risk_score": 80.0,
      "file": "credit_memo_Meta_Platforms.json",
      "summary": "The borrower Meta Platforms presents a mixed credit profile. Financial performance shows strong EBITDA ($105000.0M) but elevated leverage (0.1x).  [Re..."
    },
    {
      "id": "JPMorgan_Chase",
      "borrower_name": "JPMorgan Chase",
      "ticker": "JPM",
      "sector": "Financial",
      "report_date": "2026-02-14T02:03:52.681243",
      "risk_score": 60.0,
      "file": "credit_memo_JPMorgan_Chase.json",
      "summary": "The borrower JPMorgan Chase presents a mixed credit profile. Financial performance shows strong EBITDA ($90000.0M) but elevated leverage (10.6x).  [Re..."
    },
    {
      "id": "Goldman_Sachs",
      "borrower_name": "Goldman Sachs",
      "ticker": "GS",
      "sector": "Financial",
      "report_date": "2026-02-14T02:03:52.682158",
      "risk_score": 60.0,
      "file": "credit_memo_Goldman_Sachs.json",
      "summary": "The borrower Goldman Sachs presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: Ge..."
    },
    {
      "id": "Bank_of_America",
      "borrower_name": "Bank of America",
      "ticker": "BAC",
      "sector": "Financial",
      "report_date": "2026-02-14T02:03:52.682853",
      "risk_score": 60.0,
      "file": "credit_memo_Bank_of_America.json",
      "summary": "The borrower Bank of America presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: ..."
    }
  ],
  "credit_memos": {
    "Meta Platforms": {
      "borrower_name": "Meta Platforms",
      "report_date": "2026-02-14T02:03:52.679686",
      "executive_summary": "The borrower Meta Platforms presents a mixed credit profile. Financial performance shows strong EBITDA ($105000.0M) but elevated leverage (0.1x).  [Ref: TechCorp_10K_2025.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower Meta Platforms presents a mixed credit profile. Financial performance shows strong EBITDA ($105000.0M) but elevated leverage (0.1x).  [Ref: TechCorp_10K_2025.pdf]",
          "citations": [
            {
              "doc_id": "TechCorp_10K_2025.pdf",
              "chunk_id": "257c23be-1efc-4c74-b903-1cac04f3f4be",
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
          "content": "EBITDA: $105000.0M | Leverage: 0.1x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "key_strengths": [
        "Market leading position in core segment.",
        "Strong EBITDA generation ($105000.0M).",
        "Diversified customer base."
      ],
      "key_weaknesses": [
        "Elevated leverage at 0.1x.",
        "Exposure to cyclical end-markets.",
        "Recent management turnover."
      ],
      "mitigants": [
        "Strong free cash flow conversion.",
        "Demonstrated ability to deleverage.",
        "Sponsor support."
      ],
      "financial_ratios": {
        "leverage_ratio": 0.1,
        "dscr": 420.0,
        "current_ratio": 3.89,
        "revenue": 210000.0,
        "ebitda": 105000.0,
        "net_income": 68250.0
      },
      "historical_financials": [
        {
          "total_assets": 350000.0,
          "total_liabilities": 90000.0,
          "total_equity": 260000.0,
          "revenue": 210000.0,
          "ebitda": 105000.0,
          "net_income": 68250.0,
          "interest_expense": 250.0,
          "dscr": 420.0,
          "leverage_ratio": 0.1,
          "current_ratio": 3.89,
          "period": "FY2026"
        },
        {
          "total_assets": 322000.0,
          "total_liabilities": 82800.0,
          "total_equity": 239200.0,
          "revenue": 189000.0,
          "ebitda": 92400.0,
          "net_income": 58012.5,
          "interest_expense": 250.0,
          "dscr": 420.0,
          "leverage_ratio": 0.1,
          "current_ratio": 3.89,
          "period": "FY2025"
        },
        {
          "total_assets": 305900.0,
          "total_liabilities": 78660.0,
          "total_equity": 227240.0,
          "revenue": 173880.0,
          "ebitda": 83160.0,
          "net_income": 51051.0,
          "interest_expense": 250.0,
          "dscr": 420.0,
          "leverage_ratio": 0.1,
          "current_ratio": 3.89,
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
        "equity_value": 1087423.3140066417,
        "share_price": 185.0,
        "inputs": {
          "wacc": 0.09,
          "growth_rate": 0.03,
          "tax_rate_proxy": 0.35,
          "capex_margin": 0.05
        }
      },
      "pd_model": {
        "input_factors": {
          "Leverage Ratio": 0.1,
          "DSCR": 420.0,
          "Liquidity (Current Ratio)": 3.89,
          "EBITDA Margin": 0.5
        },
        "model_score": 80.0,
        "implied_rating": "A",
        "one_year_pd": 0.001,
        "five_year_pd": 0.01
      },
      "lgd_analysis": {
        "seniority_structure": [
          {
            "tranche": "Revolver",
            "amount": 100.0,
            "recovery_est": "70%"
          },
          {
            "tranche": "Term Loan",
            "amount": 500.0,
            "recovery_est": "60%"
          }
        ],
        "recovery_rate_assumption": 0.6499999999999999,
        "loss_given_default": 0.3500000000000001
      },
      "scenario_analysis": {
        "scenarios": [
          {
            "name": "Bear Case",
            "probability": 0.2,
            "revenue_growth": -0.05,
            "ebitda_margin": 0.2,
            "implied_share_price": 105.0
          },
          {
            "name": "Base Case",
            "probability": 0.5,
            "revenue_growth": 0.05,
            "ebitda_margin": 0.3,
            "implied_share_price": 150.0
          },
          {
            "name": "Bull Case",
            "probability": 0.3,
            "revenue_growth": 0.12,
            "ebitda_margin": 0.35,
            "implied_share_price": 210.0
          }
        ],
        "weighted_share_price": 159.0
      },
      "system_two_critique": {
        "critique_points": [
          "Credit thesis aligns with macro outlook.",
          "Valuation assumptions appear conservative."
        ],
        "conviction_score": 0.85,
        "verification_status": "PASS",
        "author_agent": "System 2"
      },
      "risk_score": 80.0,
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
          "lgd": 0.3,
          "recovery_rate": 0.7
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
          "lgd": 0.4,
          "recovery_rate": 0.6
        }
      ],
      "equity_data": {
        "market_cap": 100.0,
        "share_price": 10.0,
        "volume_avg_30d": 1000.0,
        "pe_ratio": 15.0,
        "dividend_yield": 1.0,
        "beta": 1.0
      },
      "debt_repayment_forecast": [
        {
          "year": 2026,
          "opening_balance": 520.0,
          "principal_payment": 52.0,
          "interest_payment": 26.0,
          "total_debt_service": 78.0,
          "closing_balance": 468.0
        },
        {
          "year": 2027,
          "opening_balance": 468.0,
          "principal_payment": 52.0,
          "interest_payment": 23.400000000000002,
          "total_debt_service": 75.4,
          "closing_balance": 416.0
        },
        {
          "year": 2028,
          "opening_balance": 416.0,
          "principal_payment": 52.0,
          "interest_payment": 20.8,
          "total_debt_service": 72.8,
          "closing_balance": 364.0
        },
        {
          "year": 2029,
          "opening_balance": 364.0,
          "principal_payment": 52.0,
          "interest_payment": 18.2,
          "total_debt_service": 70.2,
          "closing_balance": 312.0
        },
        {
          "year": 2030,
          "opening_balance": 312.0,
          "principal_payment": 52.0,
          "interest_payment": 15.600000000000001,
          "total_debt_service": 67.6,
          "closing_balance": 260.0
        }
      ],
      "facility_ratings": [
        {
          "facility_type": "Revolver",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.3,
          "regulatory_capital_weight": "100%"
        },
        {
          "facility_type": "Term Loan",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.4,
          "regulatory_capital_weight": "100%"
        }
      ]
    },
    "TechCorp Inc.": {
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
    "Apple Inc.": {
      "borrower_name": "Apple Inc.",
      "report_date": "2026-02-14T02:03:52.670838",
      "executive_summary": "The borrower Apple Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($155000.0M) but elevated leverage (0.9x).  [Ref: AAPL_10Q_FY25_Q1.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower Apple Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($155000.0M) but elevated leverage (0.9x).  [Ref: AAPL_10Q_FY25_Q1.pdf]",
          "citations": [
            {
              "doc_id": "AAPL_10Q_FY25_Q1.pdf",
              "chunk_id": "3f6b6aef-77f4-4f35-b04c-3faba3f07e9c",
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
          "content": "EBITDA: $155000.0M | Leverage: 0.9x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "key_strengths": [
        "Market leading position in core segment.",
        "Strong EBITDA generation ($155000.0M).",
        "Diversified customer base."
      ],
      "key_weaknesses": [
        "Elevated leverage at 0.9x.",
        "Exposure to cyclical end-markets.",
        "Recent management turnover."
      ],
      "mitigants": [
        "Strong free cash flow conversion.",
        "Demonstrated ability to deleverage.",
        "Sponsor support."
      ],
      "financial_ratios": {
        "leverage_ratio": 0.95,
        "dscr": 44.29,
        "current_ratio": 1.33,
        "revenue": 440000.0,
        "ebitda": 155000.0,
        "net_income": 100750.0
      },
      "historical_financials": [
        {
          "total_assets": 400000.0,
          "total_liabilities": 300000.0,
          "total_equity": 100000.0,
          "revenue": 440000.0,
          "ebitda": 155000.0,
          "net_income": 100750.0,
          "interest_expense": 3500.0,
          "dscr": 44.29,
          "leverage_ratio": 0.95,
          "current_ratio": 1.33,
          "period": "FY2026"
        },
        {
          "total_assets": 368000.0,
          "total_liabilities": 276000.0,
          "total_equity": 92000.0,
          "revenue": 396000.0,
          "ebitda": 136400.0,
          "net_income": 85637.5,
          "interest_expense": 3500.0,
          "dscr": 44.29,
          "leverage_ratio": 0.95,
          "current_ratio": 1.33,
          "period": "FY2025"
        },
        {
          "total_assets": 349600.0,
          "total_liabilities": 262200.0,
          "total_equity": 87400.0,
          "revenue": 364320.0,
          "ebitda": 122760.0,
          "net_income": 75361.0,
          "interest_expense": 3500.0,
          "dscr": 44.29,
          "leverage_ratio": 0.95,
          "current_ratio": 1.33,
          "period": "FY2024"
        }
      ],
      "dcf_analysis": {
        "free_cash_flow": [
          105787.5,
          110021.51874999997,
          113330.04800000001,
          115612.94231296872,
          116796.862985725
        ],
        "growth_rate": 0.03,
        "wacc": 0.09,
        "terminal_value": 2005012.8145882792,
        "enterprise_value": 1738101.082581233,
        "equity_value": 1438101.082581233,
        "share_price": 185.0,
        "inputs": {
          "wacc": 0.09,
          "growth_rate": 0.03,
          "tax_rate_proxy": 0.35,
          "capex_margin": 0.05
        }
      },
      "pd_model": {
        "input_factors": {
          "Leverage Ratio": 0.95,
          "DSCR": 44.29,
          "Liquidity (Current Ratio)": 1.33,
          "EBITDA Margin": 0.3522727272727273
        },
        "model_score": 80.0,
        "implied_rating": "A",
        "one_year_pd": 0.001,
        "five_year_pd": 0.01
      },
      "lgd_analysis": {
        "seniority_structure": [
          {
            "tranche": "Revolving Credit Facility",
            "amount": 10000.0,
            "recovery_est": "90%"
          },
          {
            "tranche": "Senior Unsecured Notes (2030)",
            "amount": 2500.0,
            "recovery_est": "70%"
          },
          {
            "tranche": "Senior Unsecured Notes (2040)",
            "amount": 1500.0,
            "recovery_est": "60%"
          },
          {
            "tranche": "Term Loan A",
            "amount": 5000.0,
            "recovery_est": "80%"
          }
        ],
        "recovery_rate_assumption": 0.75,
        "loss_given_default": 0.25
      },
      "scenario_analysis": {
        "scenarios": [
          {
            "name": "Bear Case",
            "probability": 0.2,
            "revenue_growth": -0.05,
            "ebitda_margin": 0.2,
            "implied_share_price": 105.0
          },
          {
            "name": "Base Case",
            "probability": 0.5,
            "revenue_growth": 0.05,
            "ebitda_margin": 0.3,
            "implied_share_price": 150.0
          },
          {
            "name": "Bull Case",
            "probability": 0.3,
            "revenue_growth": 0.12,
            "ebitda_margin": 0.35,
            "implied_share_price": 210.0
          }
        ],
        "weighted_share_price": 159.0
      },
      "system_two_critique": {
        "critique_points": [
          "Credit thesis aligns with macro outlook.",
          "Valuation assumptions appear conservative."
        ],
        "conviction_score": 0.85,
        "verification_status": "PASS",
        "author_agent": "System 2"
      },
      "risk_score": 80.0,
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
          "lgd": 0.1,
          "recovery_rate": 0.9
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
          "lgd": 0.3,
          "recovery_rate": 0.7
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
          "lgd": 0.4,
          "recovery_rate": 0.6
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
          "lgd": 0.2,
          "recovery_rate": 0.8
        }
      ],
      "equity_data": {
        "market_cap": 3450000.0,
        "share_price": 225.5,
        "volume_avg_30d": 45000000.0,
        "pe_ratio": 31.5,
        "dividend_yield": 0.55,
        "beta": 1.15
      },
      "debt_repayment_forecast": [
        {
          "year": 2026,
          "opening_balance": 9000.0,
          "principal_payment": 900.0,
          "interest_payment": 450.0,
          "total_debt_service": 1350.0,
          "closing_balance": 8100.0
        },
        {
          "year": 2027,
          "opening_balance": 8100.0,
          "principal_payment": 900.0,
          "interest_payment": 405.0,
          "total_debt_service": 1305.0,
          "closing_balance": 7200.0
        },
        {
          "year": 2028,
          "opening_balance": 7200.0,
          "principal_payment": 900.0,
          "interest_payment": 360.0,
          "total_debt_service": 1260.0,
          "closing_balance": 6300.0
        },
        {
          "year": 2029,
          "opening_balance": 6300.0,
          "principal_payment": 900.0,
          "interest_payment": 315.0,
          "total_debt_service": 1215.0,
          "closing_balance": 5400.0
        },
        {
          "year": 2030,
          "opening_balance": 5400.0,
          "principal_payment": 900.0,
          "interest_payment": 270.0,
          "total_debt_service": 1170.0,
          "closing_balance": 4500.0
        }
      ],
      "facility_ratings": [
        {
          "facility_type": "Revolving Credit Facility",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.1,
          "regulatory_capital_weight": "100%"
        },
        {
          "facility_type": "Senior Unsecured Notes (2030)",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.3,
          "regulatory_capital_weight": "100%"
        },
        {
          "facility_type": "Senior Unsecured Notes (2040)",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.4,
          "regulatory_capital_weight": "100%"
        },
        {
          "facility_type": "Term Loan A",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.2,
          "regulatory_capital_weight": "100%"
        }
      ]
    },
    "Alphabet Inc.": {
      "borrower_name": "Alphabet Inc.",
      "report_date": "2026-02-14T02:03:52.675494",
      "executive_summary": "The borrower Alphabet Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($138000.0M) but elevated leverage (0.0x).  [Ref: TechCorp_10K_2025.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower Alphabet Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($138000.0M) but elevated leverage (0.0x).  [Ref: TechCorp_10K_2025.pdf]",
          "citations": [
            {
              "doc_id": "TechCorp_10K_2025.pdf",
              "chunk_id": "919694d8-b379-4663-b24c-8386f7be0bc1",
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
          "content": "EBITDA: $138000.0M | Leverage: 0.0x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "key_strengths": [
        "Market leading position in core segment.",
        "Strong EBITDA generation ($138000.0M).",
        "Diversified customer base."
      ],
      "key_weaknesses": [
        "Elevated leverage at 0.0x.",
        "Exposure to cyclical end-markets.",
        "Recent management turnover."
      ],
      "mitigants": [
        "Strong free cash flow conversion.",
        "Demonstrated ability to deleverage.",
        "Sponsor support."
      ],
      "financial_ratios": {
        "leverage_ratio": 0.03,
        "dscr": 552.0,
        "current_ratio": 3.47,
        "revenue": 415000.0,
        "ebitda": 138000.0,
        "net_income": 89700.0
      },
      "historical_financials": [
        {
          "total_assets": 520000.0,
          "total_liabilities": 150000.0,
          "total_equity": 370000.0,
          "revenue": 415000.0,
          "ebitda": 138000.0,
          "net_income": 89700.0,
          "interest_expense": 250.0,
          "dscr": 552.0,
          "leverage_ratio": 0.03,
          "current_ratio": 3.47,
          "period": "FY2026"
        },
        {
          "total_assets": 478400.0,
          "total_liabilities": 138000.0,
          "total_equity": 340400.0,
          "revenue": 373500.0,
          "ebitda": 121440.0,
          "net_income": 76245.0,
          "interest_expense": 250.0,
          "dscr": 552.0,
          "leverage_ratio": 0.03,
          "current_ratio": 3.47,
          "period": "FY2025"
        },
        {
          "total_assets": 454480.0,
          "total_liabilities": 131100.0,
          "total_equity": 323380.0,
          "revenue": 343620.0,
          "ebitda": 109296.0,
          "net_income": 67095.6,
          "interest_expense": 250.0,
          "dscr": 552.0,
          "leverage_ratio": 0.03,
          "current_ratio": 3.47,
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
        "share_price": 185.0,
        "inputs": {
          "wacc": 0.09,
          "growth_rate": 0.03,
          "tax_rate_proxy": 0.35,
          "capex_margin": 0.05
        }
      },
      "pd_model": {
        "input_factors": {
          "Leverage Ratio": 0.03,
          "DSCR": 552.0,
          "Liquidity (Current Ratio)": 3.47,
          "EBITDA Margin": 0.3325301204819277
        },
        "model_score": 80.0,
        "implied_rating": "A",
        "one_year_pd": 0.001,
        "five_year_pd": 0.01
      },
      "lgd_analysis": {
        "seniority_structure": [
          {
            "tranche": "Revolver",
            "amount": 100.0,
            "recovery_est": "70%"
          },
          {
            "tranche": "Term Loan",
            "amount": 500.0,
            "recovery_est": "60%"
          }
        ],
        "recovery_rate_assumption": 0.6499999999999999,
        "loss_given_default": 0.3500000000000001
      },
      "scenario_analysis": {
        "scenarios": [
          {
            "name": "Bear Case",
            "probability": 0.2,
            "revenue_growth": -0.05,
            "ebitda_margin": 0.2,
            "implied_share_price": 105.0
          },
          {
            "name": "Base Case",
            "probability": 0.5,
            "revenue_growth": 0.05,
            "ebitda_margin": 0.3,
            "implied_share_price": 150.0
          },
          {
            "name": "Bull Case",
            "probability": 0.3,
            "revenue_growth": 0.12,
            "ebitda_margin": 0.35,
            "implied_share_price": 210.0
          }
        ],
        "weighted_share_price": 159.0
      },
      "system_two_critique": {
        "critique_points": [
          "Credit thesis aligns with macro outlook.",
          "Valuation assumptions appear conservative."
        ],
        "conviction_score": 0.85,
        "verification_status": "PASS",
        "author_agent": "System 2"
      },
      "risk_score": 80.0,
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
          "lgd": 0.3,
          "recovery_rate": 0.7
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
          "lgd": 0.4,
          "recovery_rate": 0.6
        }
      ],
      "equity_data": {
        "market_cap": 100.0,
        "share_price": 10.0,
        "volume_avg_30d": 1000.0,
        "pe_ratio": 15.0,
        "dividend_yield": 1.0,
        "beta": 1.0
      },
      "debt_repayment_forecast": [
        {
          "year": 2026,
          "opening_balance": 520.0,
          "principal_payment": 52.0,
          "interest_payment": 26.0,
          "total_debt_service": 78.0,
          "closing_balance": 468.0
        },
        {
          "year": 2027,
          "opening_balance": 468.0,
          "principal_payment": 52.0,
          "interest_payment": 23.400000000000002,
          "total_debt_service": 75.4,
          "closing_balance": 416.0
        },
        {
          "year": 2028,
          "opening_balance": 416.0,
          "principal_payment": 52.0,
          "interest_payment": 20.8,
          "total_debt_service": 72.8,
          "closing_balance": 364.0
        },
        {
          "year": 2029,
          "opening_balance": 364.0,
          "principal_payment": 52.0,
          "interest_payment": 18.2,
          "total_debt_service": 70.2,
          "closing_balance": 312.0
        },
        {
          "year": 2030,
          "opening_balance": 312.0,
          "principal_payment": 52.0,
          "interest_payment": 15.600000000000001,
          "total_debt_service": 67.6,
          "closing_balance": 260.0
        }
      ],
      "facility_ratings": [
        {
          "facility_type": "Revolver",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.3,
          "regulatory_capital_weight": "100%"
        },
        {
          "facility_type": "Term Loan",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.4,
          "regulatory_capital_weight": "100%"
        }
      ]
    },
    "NVIDIA Corp": {
      "borrower_name": "NVIDIA Corp",
      "report_date": "2026-02-14T02:03:52.674247",
      "executive_summary": "The borrower NVIDIA Corp presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: TechCorp_10K_2025.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower NVIDIA Corp presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: TechCorp_10K_2025.pdf]",
          "citations": [
            {
              "doc_id": "TechCorp_10K_2025.pdf",
              "chunk_id": "ddc5f2b8-14bf-4a9c-bea3-76b5f319137b",
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
          "content": "EBITDA: $5.0M | Leverage: 10.0x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "key_strengths": [
        "Market leading position in core segment.",
        "Strong EBITDA generation ($5.0M).",
        "Diversified customer base."
      ],
      "key_weaknesses": [
        "Elevated leverage at 10.0x.",
        "Exposure to cyclical end-markets.",
        "Recent management turnover."
      ],
      "mitigants": [
        "Strong free cash flow conversion.",
        "Demonstrated ability to deleverage.",
        "Sponsor support."
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
        "share_price": 185.0,
        "inputs": {
          "wacc": 0.09,
          "growth_rate": 0.03,
          "tax_rate_proxy": 0.35,
          "capex_margin": 0.05
        }
      },
      "pd_model": {
        "input_factors": {
          "Leverage Ratio": 10.0,
          "DSCR": 5.0,
          "Liquidity (Current Ratio)": 2.0,
          "EBITDA Margin": 0.25
        },
        "model_score": 60.0,
        "implied_rating": "BB",
        "one_year_pd": 0.02,
        "five_year_pd": 0.1
      },
      "lgd_analysis": {
        "seniority_structure": [
          {
            "tranche": "Revolver",
            "amount": 100.0,
            "recovery_est": "70%"
          },
          {
            "tranche": "Term Loan",
            "amount": 500.0,
            "recovery_est": "60%"
          }
        ],
        "recovery_rate_assumption": 0.6499999999999999,
        "loss_given_default": 0.3500000000000001
      },
      "scenario_analysis": {
        "scenarios": [
          {
            "name": "Bear Case",
            "probability": 0.2,
            "revenue_growth": -0.05,
            "ebitda_margin": 0.2,
            "implied_share_price": 105.0
          },
          {
            "name": "Base Case",
            "probability": 0.5,
            "revenue_growth": 0.05,
            "ebitda_margin": 0.3,
            "implied_share_price": 150.0
          },
          {
            "name": "Bull Case",
            "probability": 0.3,
            "revenue_growth": 0.12,
            "ebitda_margin": 0.35,
            "implied_share_price": 210.0
          }
        ],
        "weighted_share_price": 159.0
      },
      "system_two_critique": {
        "critique_points": [
          "Credit thesis aligns with macro outlook.",
          "Valuation assumptions appear conservative."
        ],
        "conviction_score": 0.85,
        "verification_status": "PASS",
        "author_agent": "System 2"
      },
      "risk_score": 60.0,
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
          "lgd": 0.3,
          "recovery_rate": 0.7
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
          "lgd": 0.4,
          "recovery_rate": 0.6
        }
      ],
      "equity_data": {
        "market_cap": 100.0,
        "share_price": 10.0,
        "volume_avg_30d": 1000.0,
        "pe_ratio": 15.0,
        "dividend_yield": 1.0,
        "beta": 1.0
      },
      "debt_repayment_forecast": [
        {
          "year": 2026,
          "opening_balance": 520.0,
          "principal_payment": 52.0,
          "interest_payment": 26.0,
          "total_debt_service": 78.0,
          "closing_balance": 468.0
        },
        {
          "year": 2027,
          "opening_balance": 468.0,
          "principal_payment": 52.0,
          "interest_payment": 23.400000000000002,
          "total_debt_service": 75.4,
          "closing_balance": 416.0
        },
        {
          "year": 2028,
          "opening_balance": 416.0,
          "principal_payment": 52.0,
          "interest_payment": 20.8,
          "total_debt_service": 72.8,
          "closing_balance": 364.0
        },
        {
          "year": 2029,
          "opening_balance": 364.0,
          "principal_payment": 52.0,
          "interest_payment": 18.2,
          "total_debt_service": 70.2,
          "closing_balance": 312.0
        },
        {
          "year": 2030,
          "opening_balance": 312.0,
          "principal_payment": 52.0,
          "interest_payment": 15.600000000000001,
          "total_debt_service": 67.6,
          "closing_balance": 260.0
        }
      ],
      "facility_ratings": [
        {
          "facility_type": "Revolver",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.3,
          "regulatory_capital_weight": "100%"
        },
        {
          "facility_type": "Term Loan",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.4,
          "regulatory_capital_weight": "100%"
        }
      ]
    },
    "Bank of America": {
      "borrower_name": "Bank of America",
      "report_date": "2026-02-14T02:03:52.682853",
      "executive_summary": "The borrower Bank of America presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: Generic_Borrower_Profile.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower Bank of America presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: Generic_Borrower_Profile.pdf]",
          "citations": [
            {
              "doc_id": "Generic_Borrower_Profile.pdf",
              "chunk_id": "a8eff95b-561b-491c-b86e-cd96005db27f",
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
      "key_strengths": [
        "Market leading position in core segment.",
        "Strong EBITDA generation ($5.0M).",
        "Diversified customer base."
      ],
      "key_weaknesses": [
        "Elevated leverage at 10.0x.",
        "Exposure to cyclical end-markets.",
        "Recent management turnover."
      ],
      "mitigants": [
        "Strong free cash flow conversion.",
        "Demonstrated ability to deleverage.",
        "Sponsor support."
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
        "share_price": 185.0,
        "inputs": {
          "wacc": 0.09,
          "growth_rate": 0.03,
          "tax_rate_proxy": 0.35,
          "capex_margin": 0.05
        }
      },
      "pd_model": {
        "input_factors": {
          "Leverage Ratio": 10.0,
          "DSCR": 5.0,
          "Liquidity (Current Ratio)": 2.0,
          "EBITDA Margin": 0.25
        },
        "model_score": 60.0,
        "implied_rating": "BB",
        "one_year_pd": 0.02,
        "five_year_pd": 0.1
      },
      "lgd_analysis": {
        "seniority_structure": [
          {
            "tranche": "Revolver",
            "amount": 100.0,
            "recovery_est": "70%"
          },
          {
            "tranche": "Term Loan",
            "amount": 500.0,
            "recovery_est": "60%"
          }
        ],
        "recovery_rate_assumption": 0.6499999999999999,
        "loss_given_default": 0.3500000000000001
      },
      "scenario_analysis": {
        "scenarios": [
          {
            "name": "Bear Case",
            "probability": 0.2,
            "revenue_growth": -0.05,
            "ebitda_margin": 0.2,
            "implied_share_price": 105.0
          },
          {
            "name": "Base Case",
            "probability": 0.5,
            "revenue_growth": 0.05,
            "ebitda_margin": 0.3,
            "implied_share_price": 150.0
          },
          {
            "name": "Bull Case",
            "probability": 0.3,
            "revenue_growth": 0.12,
            "ebitda_margin": 0.35,
            "implied_share_price": 210.0
          }
        ],
        "weighted_share_price": 159.0
      },
      "system_two_critique": {
        "critique_points": [
          "Credit thesis aligns with macro outlook.",
          "Valuation assumptions appear conservative."
        ],
        "conviction_score": 0.85,
        "verification_status": "PASS",
        "author_agent": "System 2"
      },
      "risk_score": 60.0,
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
          "lgd": 0.3,
          "recovery_rate": 0.7
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
          "lgd": 0.4,
          "recovery_rate": 0.6
        }
      ],
      "equity_data": {
        "market_cap": 100.0,
        "share_price": 10.0,
        "volume_avg_30d": 1000.0,
        "pe_ratio": 15.0,
        "dividend_yield": 1.0,
        "beta": 1.0
      },
      "debt_repayment_forecast": [
        {
          "year": 2026,
          "opening_balance": 520.0,
          "principal_payment": 52.0,
          "interest_payment": 26.0,
          "total_debt_service": 78.0,
          "closing_balance": 468.0
        },
        {
          "year": 2027,
          "opening_balance": 468.0,
          "principal_payment": 52.0,
          "interest_payment": 23.400000000000002,
          "total_debt_service": 75.4,
          "closing_balance": 416.0
        },
        {
          "year": 2028,
          "opening_balance": 416.0,
          "principal_payment": 52.0,
          "interest_payment": 20.8,
          "total_debt_service": 72.8,
          "closing_balance": 364.0
        },
        {
          "year": 2029,
          "opening_balance": 364.0,
          "principal_payment": 52.0,
          "interest_payment": 18.2,
          "total_debt_service": 70.2,
          "closing_balance": 312.0
        },
        {
          "year": 2030,
          "opening_balance": 312.0,
          "principal_payment": 52.0,
          "interest_payment": 15.600000000000001,
          "total_debt_service": 67.6,
          "closing_balance": 260.0
        }
      ],
      "facility_ratings": [
        {
          "facility_type": "Revolver",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.3,
          "regulatory_capital_weight": "100%"
        },
        {
          "facility_type": "Term Loan",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.4,
          "regulatory_capital_weight": "100%"
        }
      ]
    },
    "JPMorgan Chase": {
      "borrower_name": "JPMorgan Chase",
      "report_date": "2026-02-14T02:03:52.681243",
      "executive_summary": "The borrower JPMorgan Chase presents a mixed credit profile. Financial performance shows strong EBITDA ($90000.0M) but elevated leverage (10.6x).  [Ref: JPM_Earnings_Release_4Q24.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower JPMorgan Chase presents a mixed credit profile. Financial performance shows strong EBITDA ($90000.0M) but elevated leverage (10.6x).  [Ref: JPM_Earnings_Release_4Q24.pdf]",
          "citations": [
            {
              "doc_id": "JPM_Earnings_Release_4Q24.pdf",
              "chunk_id": "39edad80-0ea6-454e-8006-d82c0c0c2cc3",
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
          "content": "EBITDA: $90000.0M | Leverage: 10.6x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "key_strengths": [
        "Market leading position in core segment.",
        "Strong EBITDA generation ($90000.0M).",
        "Diversified customer base."
      ],
      "key_weaknesses": [
        "Elevated leverage at 10.6x.",
        "Exposure to cyclical end-markets.",
        "Recent management turnover."
      ],
      "mitigants": [
        "Strong free cash flow conversion.",
        "Demonstrated ability to deleverage.",
        "Sponsor support."
      ],
      "financial_ratios": {
        "leverage_ratio": 10.594202898550725,
        "dscr": 999.0,
        "current_ratio": 1.1,
        "revenue": 170000.0,
        "ebitda": 90000.0,
        "net_income": 56000.0
      },
      "historical_financials": [
        {
          "total_assets": 4000000.0,
          "total_liabilities": 3655000.0,
          "total_equity": 345000.0,
          "revenue": 170000.0,
          "ebitda": 90000.0,
          "net_income": 56000.0,
          "interest_expense": 1.0,
          "dscr": 999.0,
          "leverage_ratio": 10.594202898550725,
          "current_ratio": 1.1,
          "period": "FY2024 Q4"
        },
        {
          "total_assets": 3680000.0,
          "total_liabilities": 3362600.0,
          "total_equity": 317400.0,
          "revenue": 153000.0,
          "ebitda": 79200.0,
          "net_income": 47600.0,
          "interest_expense": 1.0,
          "dscr": 999.0,
          "leverage_ratio": 10.594202898550725,
          "current_ratio": 1.1,
          "period": "FY2023"
        },
        {
          "total_assets": 3496000.0,
          "total_liabilities": 3194470.0,
          "total_equity": 301530.0,
          "revenue": 140760.0,
          "ebitda": 71280.0,
          "net_income": 41888.0,
          "interest_expense": 1.0,
          "dscr": 999.0,
          "leverage_ratio": 10.594202898550725,
          "current_ratio": 1.1,
          "period": "FY2022"
        }
      ],
      "dcf_analysis": {
        "free_cash_flow": [
          61425.0,
          63883.46249999999,
          65804.54400000001,
          67130.09553656248,
          67817.53334655
        ],
        "growth_rate": 0.03,
        "wacc": 0.09,
        "terminal_value": 1164200.989115775,
        "enterprise_value": 1009219.9834342643,
        "equity_value": -2645780.0165657355,
        "share_price": -2645780.0165657355,
        "inputs": {
          "wacc": 0.09,
          "growth_rate": 0.03,
          "tax_rate_proxy": 0.35,
          "capex_margin": 0.05
        }
      },
      "pd_model": {
        "input_factors": {
          "Leverage Ratio": 10.594202898550725,
          "DSCR": 999.0,
          "Liquidity (Current Ratio)": 1.1,
          "EBITDA Margin": 0.5294117647058824
        },
        "model_score": 60.0,
        "implied_rating": "BB",
        "one_year_pd": 0.02,
        "five_year_pd": 0.1
      },
      "lgd_analysis": {
        "seniority_structure": [
          {
            "tranche": "Revolver",
            "amount": 100.0,
            "recovery_est": "70%"
          },
          {
            "tranche": "Term Loan",
            "amount": 500.0,
            "recovery_est": "60%"
          }
        ],
        "recovery_rate_assumption": 0.6499999999999999,
        "loss_given_default": 0.3500000000000001
      },
      "scenario_analysis": {
        "scenarios": [
          {
            "name": "Bear Case",
            "probability": 0.2,
            "revenue_growth": -0.05,
            "ebitda_margin": 0.2,
            "implied_share_price": 105.0
          },
          {
            "name": "Base Case",
            "probability": 0.5,
            "revenue_growth": 0.05,
            "ebitda_margin": 0.3,
            "implied_share_price": 150.0
          },
          {
            "name": "Bull Case",
            "probability": 0.3,
            "revenue_growth": 0.12,
            "ebitda_margin": 0.35,
            "implied_share_price": 210.0
          }
        ],
        "weighted_share_price": 159.0
      },
      "system_two_critique": {
        "critique_points": [
          "Credit thesis aligns with macro outlook.",
          "Valuation assumptions appear conservative."
        ],
        "conviction_score": 0.85,
        "verification_status": "PASS",
        "author_agent": "System 2"
      },
      "risk_score": 60.0,
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
          "lgd": 0.3,
          "recovery_rate": 0.7
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
          "lgd": 0.4,
          "recovery_rate": 0.6
        }
      ],
      "equity_data": {
        "market_cap": 580000.0,
        "share_price": 205.1,
        "volume_avg_30d": 9500000.0,
        "pe_ratio": 11.8,
        "dividend_yield": 2.3,
        "beta": 1.05
      },
      "debt_repayment_forecast": [
        {
          "year": 2026,
          "opening_balance": 520.0,
          "principal_payment": 52.0,
          "interest_payment": 26.0,
          "total_debt_service": 78.0,
          "closing_balance": 468.0
        },
        {
          "year": 2027,
          "opening_balance": 468.0,
          "principal_payment": 52.0,
          "interest_payment": 23.400000000000002,
          "total_debt_service": 75.4,
          "closing_balance": 416.0
        },
        {
          "year": 2028,
          "opening_balance": 416.0,
          "principal_payment": 52.0,
          "interest_payment": 20.8,
          "total_debt_service": 72.8,
          "closing_balance": 364.0
        },
        {
          "year": 2029,
          "opening_balance": 364.0,
          "principal_payment": 52.0,
          "interest_payment": 18.2,
          "total_debt_service": 70.2,
          "closing_balance": 312.0
        },
        {
          "year": 2030,
          "opening_balance": 312.0,
          "principal_payment": 52.0,
          "interest_payment": 15.600000000000001,
          "total_debt_service": 67.6,
          "closing_balance": 260.0
        }
      ],
      "facility_ratings": [
        {
          "facility_type": "Revolver",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.3,
          "regulatory_capital_weight": "100%"
        },
        {
          "facility_type": "Term Loan",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.4,
          "regulatory_capital_weight": "100%"
        }
      ]
    },
    "Microsoft Corp": {
      "borrower_name": "Microsoft Corp",
      "report_date": "2026-02-14T02:03:52.672795",
      "executive_summary": "The borrower Microsoft Corp presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: TechCorp_10K_2025.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower Microsoft Corp presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: TechCorp_10K_2025.pdf]",
          "citations": [
            {
              "doc_id": "TechCorp_10K_2025.pdf",
              "chunk_id": "5a37462d-4831-471d-b83b-f2efc713fb90",
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
          "content": "EBITDA: $5.0M | Leverage: 10.0x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "key_strengths": [
        "Market leading position in core segment.",
        "Strong EBITDA generation ($5.0M).",
        "Diversified customer base."
      ],
      "key_weaknesses": [
        "Elevated leverage at 10.0x.",
        "Exposure to cyclical end-markets.",
        "Recent management turnover."
      ],
      "mitigants": [
        "Strong free cash flow conversion.",
        "Demonstrated ability to deleverage.",
        "Sponsor support."
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
        "share_price": 185.0,
        "inputs": {
          "wacc": 0.09,
          "growth_rate": 0.03,
          "tax_rate_proxy": 0.35,
          "capex_margin": 0.05
        }
      },
      "pd_model": {
        "input_factors": {
          "Leverage Ratio": 10.0,
          "DSCR": 5.0,
          "Liquidity (Current Ratio)": 2.0,
          "EBITDA Margin": 0.25
        },
        "model_score": 60.0,
        "implied_rating": "BB",
        "one_year_pd": 0.02,
        "five_year_pd": 0.1
      },
      "lgd_analysis": {
        "seniority_structure": [
          {
            "tranche": "Revolver",
            "amount": 100.0,
            "recovery_est": "70%"
          },
          {
            "tranche": "Term Loan",
            "amount": 500.0,
            "recovery_est": "60%"
          }
        ],
        "recovery_rate_assumption": 0.6499999999999999,
        "loss_given_default": 0.3500000000000001
      },
      "scenario_analysis": {
        "scenarios": [
          {
            "name": "Bear Case",
            "probability": 0.2,
            "revenue_growth": -0.05,
            "ebitda_margin": 0.2,
            "implied_share_price": 105.0
          },
          {
            "name": "Base Case",
            "probability": 0.5,
            "revenue_growth": 0.05,
            "ebitda_margin": 0.3,
            "implied_share_price": 150.0
          },
          {
            "name": "Bull Case",
            "probability": 0.3,
            "revenue_growth": 0.12,
            "ebitda_margin": 0.35,
            "implied_share_price": 210.0
          }
        ],
        "weighted_share_price": 159.0
      },
      "system_two_critique": {
        "critique_points": [
          "Credit thesis aligns with macro outlook.",
          "Valuation assumptions appear conservative."
        ],
        "conviction_score": 0.85,
        "verification_status": "PASS",
        "author_agent": "System 2"
      },
      "risk_score": 60.0,
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
          "lgd": 0.3,
          "recovery_rate": 0.7
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
          "lgd": 0.4,
          "recovery_rate": 0.6
        }
      ],
      "equity_data": {
        "market_cap": 100.0,
        "share_price": 10.0,
        "volume_avg_30d": 1000.0,
        "pe_ratio": 15.0,
        "dividend_yield": 1.0,
        "beta": 1.0
      },
      "debt_repayment_forecast": [
        {
          "year": 2026,
          "opening_balance": 520.0,
          "principal_payment": 52.0,
          "interest_payment": 26.0,
          "total_debt_service": 78.0,
          "closing_balance": 468.0
        },
        {
          "year": 2027,
          "opening_balance": 468.0,
          "principal_payment": 52.0,
          "interest_payment": 23.400000000000002,
          "total_debt_service": 75.4,
          "closing_balance": 416.0
        },
        {
          "year": 2028,
          "opening_balance": 416.0,
          "principal_payment": 52.0,
          "interest_payment": 20.8,
          "total_debt_service": 72.8,
          "closing_balance": 364.0
        },
        {
          "year": 2029,
          "opening_balance": 364.0,
          "principal_payment": 52.0,
          "interest_payment": 18.2,
          "total_debt_service": 70.2,
          "closing_balance": 312.0
        },
        {
          "year": 2030,
          "opening_balance": 312.0,
          "principal_payment": 52.0,
          "interest_payment": 15.600000000000001,
          "total_debt_service": 67.6,
          "closing_balance": 260.0
        }
      ],
      "facility_ratings": [
        {
          "facility_type": "Revolver",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.3,
          "regulatory_capital_weight": "100%"
        },
        {
          "facility_type": "Term Loan",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.4,
          "regulatory_capital_weight": "100%"
        }
      ]
    },
    "Goldman Sachs": {
      "borrower_name": "Goldman Sachs",
      "report_date": "2026-02-14T02:03:52.682158",
      "executive_summary": "The borrower Goldman Sachs presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: Generic_Borrower_Profile.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower Goldman Sachs presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: Generic_Borrower_Profile.pdf]",
          "citations": [
            {
              "doc_id": "Generic_Borrower_Profile.pdf",
              "chunk_id": "5d86e369-792e-4294-9284-ada8cbbee80a",
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
      "key_strengths": [
        "Market leading position in core segment.",
        "Strong EBITDA generation ($5.0M).",
        "Diversified customer base."
      ],
      "key_weaknesses": [
        "Elevated leverage at 10.0x.",
        "Exposure to cyclical end-markets.",
        "Recent management turnover."
      ],
      "mitigants": [
        "Strong free cash flow conversion.",
        "Demonstrated ability to deleverage.",
        "Sponsor support."
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
        "share_price": 185.0,
        "inputs": {
          "wacc": 0.09,
          "growth_rate": 0.03,
          "tax_rate_proxy": 0.35,
          "capex_margin": 0.05
        }
      },
      "pd_model": {
        "input_factors": {
          "Leverage Ratio": 10.0,
          "DSCR": 5.0,
          "Liquidity (Current Ratio)": 2.0,
          "EBITDA Margin": 0.25
        },
        "model_score": 60.0,
        "implied_rating": "BB",
        "one_year_pd": 0.02,
        "five_year_pd": 0.1
      },
      "lgd_analysis": {
        "seniority_structure": [
          {
            "tranche": "Revolver",
            "amount": 100.0,
            "recovery_est": "70%"
          },
          {
            "tranche": "Term Loan",
            "amount": 500.0,
            "recovery_est": "60%"
          }
        ],
        "recovery_rate_assumption": 0.6499999999999999,
        "loss_given_default": 0.3500000000000001
      },
      "scenario_analysis": {
        "scenarios": [
          {
            "name": "Bear Case",
            "probability": 0.2,
            "revenue_growth": -0.05,
            "ebitda_margin": 0.2,
            "implied_share_price": 105.0
          },
          {
            "name": "Base Case",
            "probability": 0.5,
            "revenue_growth": 0.05,
            "ebitda_margin": 0.3,
            "implied_share_price": 150.0
          },
          {
            "name": "Bull Case",
            "probability": 0.3,
            "revenue_growth": 0.12,
            "ebitda_margin": 0.35,
            "implied_share_price": 210.0
          }
        ],
        "weighted_share_price": 159.0
      },
      "system_two_critique": {
        "critique_points": [
          "Credit thesis aligns with macro outlook.",
          "Valuation assumptions appear conservative."
        ],
        "conviction_score": 0.85,
        "verification_status": "PASS",
        "author_agent": "System 2"
      },
      "risk_score": 60.0,
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
          "lgd": 0.3,
          "recovery_rate": 0.7
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
          "lgd": 0.4,
          "recovery_rate": 0.6
        }
      ],
      "equity_data": {
        "market_cap": 100.0,
        "share_price": 10.0,
        "volume_avg_30d": 1000.0,
        "pe_ratio": 15.0,
        "dividend_yield": 1.0,
        "beta": 1.0
      },
      "debt_repayment_forecast": [
        {
          "year": 2026,
          "opening_balance": 520.0,
          "principal_payment": 52.0,
          "interest_payment": 26.0,
          "total_debt_service": 78.0,
          "closing_balance": 468.0
        },
        {
          "year": 2027,
          "opening_balance": 468.0,
          "principal_payment": 52.0,
          "interest_payment": 23.400000000000002,
          "total_debt_service": 75.4,
          "closing_balance": 416.0
        },
        {
          "year": 2028,
          "opening_balance": 416.0,
          "principal_payment": 52.0,
          "interest_payment": 20.8,
          "total_debt_service": 72.8,
          "closing_balance": 364.0
        },
        {
          "year": 2029,
          "opening_balance": 364.0,
          "principal_payment": 52.0,
          "interest_payment": 18.2,
          "total_debt_service": 70.2,
          "closing_balance": 312.0
        },
        {
          "year": 2030,
          "opening_balance": 312.0,
          "principal_payment": 52.0,
          "interest_payment": 15.600000000000001,
          "total_debt_service": 67.6,
          "closing_balance": 260.0
        }
      ],
      "facility_ratings": [
        {
          "facility_type": "Revolver",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.3,
          "regulatory_capital_weight": "100%"
        },
        {
          "facility_type": "Term Loan",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.4,
          "regulatory_capital_weight": "100%"
        }
      ]
    },
    "Tesla Inc.": {
      "borrower_name": "Tesla Inc.",
      "report_date": "2026-02-14T02:03:52.678370",
      "executive_summary": "The borrower Tesla Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($12500.0M) but elevated leverage (3.7x).  [Ref: TSLA_10Q_FY24_Q3.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower Tesla Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($12500.0M) but elevated leverage (3.7x).  [Ref: TSLA_10Q_FY24_Q3.pdf]",
          "citations": [
            {
              "doc_id": "TSLA_10Q_FY24_Q3.pdf",
              "chunk_id": "9b9ac209-7762-449a-b779-a5698ed396c6",
              "page_number": 8
            }
          ],
          "author_agent": "Writer"
        },
        {
          "title": "Key Risks & Mitigants",
          "content": "Key Risks:\n- CONNECTED PARTY RISK: X (Twitter) via Tesla Inc. -> X (Twitter) is High risk.\n- CONNECTED PARTY RISK: xAI via Tesla Inc. -> xAI is High risk.\n",
          "citations": [],
          "author_agent": "Risk Officer"
        },
        {
          "title": "Financial Analysis",
          "content": "EBITDA: $12500.0M | Leverage: 3.7x",
          "citations": [],
          "author_agent": "Quant"
        }
      ],
      "key_strengths": [
        "Market leading position in core segment.",
        "Strong EBITDA generation ($12500.0M).",
        "Diversified customer base."
      ],
      "key_weaknesses": [
        "Elevated leverage at 3.7x.",
        "Exposure to cyclical end-markets.",
        "Recent management turnover."
      ],
      "mitigants": [
        "Strong free cash flow conversion.",
        "Demonstrated ability to deleverage.",
        "Sponsor support."
      ],
      "financial_ratios": {
        "leverage_ratio": 3.656,
        "dscr": 83.33333333333333,
        "current_ratio": 1.7,
        "revenue": 97000.0,
        "ebitda": 12500.0,
        "net_income": 8400.0
      },
      "historical_financials": [
        {
          "total_assets": 119800.0,
          "total_liabilities": 45700.0,
          "total_equity": 74100.0,
          "revenue": 97000.0,
          "ebitda": 12500.0,
          "net_income": 8400.0,
          "interest_expense": 150.0,
          "dscr": 83.33333333333333,
          "leverage_ratio": 3.656,
          "current_ratio": 1.7,
          "period": "FY2024 Q3"
        },
        {
          "total_assets": 110216.0,
          "total_liabilities": 42044.0,
          "total_equity": 68172.0,
          "revenue": 87300.0,
          "ebitda": 11000.0,
          "net_income": 7140.0,
          "interest_expense": 150.0,
          "dscr": 83.33333333333333,
          "leverage_ratio": 3.656,
          "current_ratio": 1.7,
          "period": "FY2023"
        },
        {
          "total_assets": 104705.2,
          "total_liabilities": 39941.799999999996,
          "total_equity": 64763.399999999994,
          "revenue": 80316.0,
          "ebitda": 9900.0,
          "net_income": 6283.2,
          "interest_expense": 150.0,
          "dscr": 83.33333333333333,
          "leverage_ratio": 3.656,
          "current_ratio": 1.7,
          "period": "FY2022"
        }
      ],
      "dcf_analysis": {
        "free_cash_flow": [
          8531.25,
          8872.703124999998,
          9139.52,
          9323.624380078123,
          9419.1018536875
        ],
        "growth_rate": 0.03,
        "wacc": 0.09,
        "terminal_value": 161694.58182163542,
        "enterprise_value": 140169.44214364782,
        "equity_value": 94469.44214364782,
        "share_price": 185.0,
        "inputs": {
          "wacc": 0.09,
          "growth_rate": 0.03,
          "tax_rate_proxy": 0.35,
          "capex_margin": 0.05
        }
      },
      "pd_model": {
        "input_factors": {
          "Leverage Ratio": 3.656,
          "DSCR": 83.33333333333333,
          "Liquidity (Current Ratio)": 1.7,
          "EBITDA Margin": 0.12886597938144329
        },
        "model_score": 70.0,
        "implied_rating": "BBB",
        "one_year_pd": 0.005,
        "five_year_pd": 0.03
      },
      "lgd_analysis": {
        "seniority_structure": [
          {
            "tranche": "ABL Revolver",
            "amount": 5000.0,
            "recovery_est": "85%"
          },
          {
            "tranche": "Convertible Senior Notes",
            "amount": 1800.0,
            "recovery_est": "50%"
          },
          {
            "tranche": "Auto ABS Facilities",
            "amount": 3000.0,
            "recovery_est": "95%"
          }
        ],
        "recovery_rate_assumption": 0.7666666666666666,
        "loss_given_default": 0.2333333333333334
      },
      "scenario_analysis": {
        "scenarios": [
          {
            "name": "Bear Case",
            "probability": 0.2,
            "revenue_growth": -0.05,
            "ebitda_margin": 0.2,
            "implied_share_price": 105.0
          },
          {
            "name": "Base Case",
            "probability": 0.5,
            "revenue_growth": 0.05,
            "ebitda_margin": 0.3,
            "implied_share_price": 150.0
          },
          {
            "name": "Bull Case",
            "probability": 0.3,
            "revenue_growth": 0.12,
            "ebitda_margin": 0.35,
            "implied_share_price": 210.0
          }
        ],
        "weighted_share_price": 159.0
      },
      "system_two_critique": {
        "critique_points": [
          "Credit thesis aligns with macro outlook.",
          "Valuation assumptions appear conservative."
        ],
        "conviction_score": 0.85,
        "verification_status": "PASS",
        "author_agent": "System 2"
      },
      "risk_score": 70.0,
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
          "lgd": 0.15,
          "recovery_rate": 0.85
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
          "lgd": 0.5,
          "recovery_rate": 0.5
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
          "lgd": 0.05,
          "recovery_rate": 0.95
        }
      ],
      "equity_data": {
        "market_cap": 850000.0,
        "share_price": 265.4,
        "volume_avg_30d": 98000000.0,
        "pe_ratio": 68.2,
        "dividend_yield": 0.0,
        "beta": 2.05
      },
      "debt_repayment_forecast": [
        {
          "year": 2026,
          "opening_balance": 4500.0,
          "principal_payment": 450.0,
          "interest_payment": 225.0,
          "total_debt_service": 675.0,
          "closing_balance": 4050.0
        },
        {
          "year": 2027,
          "opening_balance": 4050.0,
          "principal_payment": 450.0,
          "interest_payment": 202.5,
          "total_debt_service": 652.5,
          "closing_balance": 3600.0
        },
        {
          "year": 2028,
          "opening_balance": 3600.0,
          "principal_payment": 450.0,
          "interest_payment": 180.0,
          "total_debt_service": 630.0,
          "closing_balance": 3150.0
        },
        {
          "year": 2029,
          "opening_balance": 3150.0,
          "principal_payment": 450.0,
          "interest_payment": 157.5,
          "total_debt_service": 607.5,
          "closing_balance": 2700.0
        },
        {
          "year": 2030,
          "opening_balance": 2700.0,
          "principal_payment": 450.0,
          "interest_payment": 135.0,
          "total_debt_service": 585.0,
          "closing_balance": 2250.0
        }
      ],
      "facility_ratings": [
        {
          "facility_type": "ABL Revolver",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.15,
          "regulatory_capital_weight": "100%"
        },
        {
          "facility_type": "Convertible Senior Notes",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.5,
          "regulatory_capital_weight": "100%"
        },
        {
          "facility_type": "Auto ABS Facilities",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.05,
          "regulatory_capital_weight": "100%"
        }
      ]
    },
    "Amazon.com Inc.": {
      "borrower_name": "Amazon.com Inc.",
      "report_date": "2026-02-14T02:03:52.676637",
      "executive_summary": "The borrower Amazon.com Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: Generic_Borrower_Profile.pdf]",
      "sections": [
        {
          "title": "Executive Summary",
          "content": "The borrower Amazon.com Inc. presents a mixed credit profile. Financial performance shows strong EBITDA ($5.0M) but elevated leverage (10.0x).  [Ref: Generic_Borrower_Profile.pdf]",
          "citations": [
            {
              "doc_id": "Generic_Borrower_Profile.pdf",
              "chunk_id": "5f469ec3-abf2-4d4c-8f34-84371e67cad6",
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
      "key_strengths": [
        "Market leading position in core segment.",
        "Strong EBITDA generation ($5.0M).",
        "Diversified customer base."
      ],
      "key_weaknesses": [
        "Elevated leverage at 10.0x.",
        "Exposure to cyclical end-markets.",
        "Recent management turnover."
      ],
      "mitigants": [
        "Strong free cash flow conversion.",
        "Demonstrated ability to deleverage.",
        "Sponsor support."
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
        "share_price": 185.0,
        "inputs": {
          "wacc": 0.09,
          "growth_rate": 0.03,
          "tax_rate_proxy": 0.35,
          "capex_margin": 0.05
        }
      },
      "pd_model": {
        "input_factors": {
          "Leverage Ratio": 10.0,
          "DSCR": 5.0,
          "Liquidity (Current Ratio)": 2.0,
          "EBITDA Margin": 0.25
        },
        "model_score": 60.0,
        "implied_rating": "BB",
        "one_year_pd": 0.02,
        "five_year_pd": 0.1
      },
      "lgd_analysis": {
        "seniority_structure": [
          {
            "tranche": "Revolver",
            "amount": 100.0,
            "recovery_est": "70%"
          },
          {
            "tranche": "Term Loan",
            "amount": 500.0,
            "recovery_est": "60%"
          }
        ],
        "recovery_rate_assumption": 0.6499999999999999,
        "loss_given_default": 0.3500000000000001
      },
      "scenario_analysis": {
        "scenarios": [
          {
            "name": "Bear Case",
            "probability": 0.2,
            "revenue_growth": -0.05,
            "ebitda_margin": 0.2,
            "implied_share_price": 105.0
          },
          {
            "name": "Base Case",
            "probability": 0.5,
            "revenue_growth": 0.05,
            "ebitda_margin": 0.3,
            "implied_share_price": 150.0
          },
          {
            "name": "Bull Case",
            "probability": 0.3,
            "revenue_growth": 0.12,
            "ebitda_margin": 0.35,
            "implied_share_price": 210.0
          }
        ],
        "weighted_share_price": 159.0
      },
      "system_two_critique": {
        "critique_points": [
          "Credit thesis aligns with macro outlook.",
          "Valuation assumptions appear conservative."
        ],
        "conviction_score": 0.85,
        "verification_status": "PASS",
        "author_agent": "System 2"
      },
      "risk_score": 60.0,
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
          "lgd": 0.3,
          "recovery_rate": 0.7
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
          "lgd": 0.4,
          "recovery_rate": 0.6
        }
      ],
      "equity_data": {
        "market_cap": 100.0,
        "share_price": 10.0,
        "volume_avg_30d": 1000.0,
        "pe_ratio": 15.0,
        "dividend_yield": 1.0,
        "beta": 1.0
      },
      "debt_repayment_forecast": [
        {
          "year": 2026,
          "opening_balance": 520.0,
          "principal_payment": 52.0,
          "interest_payment": 26.0,
          "total_debt_service": 78.0,
          "closing_balance": 468.0
        },
        {
          "year": 2027,
          "opening_balance": 468.0,
          "principal_payment": 52.0,
          "interest_payment": 23.400000000000002,
          "total_debt_service": 75.4,
          "closing_balance": 416.0
        },
        {
          "year": 2028,
          "opening_balance": 416.0,
          "principal_payment": 52.0,
          "interest_payment": 20.8,
          "total_debt_service": 72.8,
          "closing_balance": 364.0
        },
        {
          "year": 2029,
          "opening_balance": 364.0,
          "principal_payment": 52.0,
          "interest_payment": 18.2,
          "total_debt_service": 70.2,
          "closing_balance": 312.0
        },
        {
          "year": 2030,
          "opening_balance": 312.0,
          "principal_payment": 52.0,
          "interest_payment": 15.600000000000001,
          "total_debt_service": 67.6,
          "closing_balance": 260.0
        }
      ],
      "facility_ratings": [
        {
          "facility_type": "Revolver",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.3,
          "regulatory_capital_weight": "100%"
        },
        {
          "facility_type": "Term Loan",
          "snc_rating": "Pass",
          "pd_1y": 0.015,
          "lgd": 0.4,
          "regulatory_capital_weight": "100%"
        }
      ]
    }
  },
  "market_data": {
    "tickers": [
      "AAPL",
      "AMZN",
      "GOOGL",
      "META",
      "MSFT",
      "NVDA",
      "TSLA"
    ],
    "financials_db_snapshot": {
      "AAPL": {
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "history": [
          {
            "fiscal_year": 2021,
            "revenue": 365817,
            "ebitda": 120233,
            "total_debt": 124719,
            "cash_equivalents": 34940,
            "interest_expense": 2645,
            "total_assets": 351002,
            "total_liabilities": 287912,
            "total_equity": 63090
          },
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
            "revenue": 391035,
            "ebitda": 129629,
            "total_debt": 106000,
            "cash_equivalents": 29965,
            "interest_expense": 3900,
            "total_assets": 360000,
            "total_liabilities": 290000,
            "total_equity": 70000
          },
          {
            "fiscal_year": 2025,
            "revenue": 415000,
            "ebitda": 142000,
            "total_debt": 100000,
            "cash_equivalents": 35000,
            "interest_expense": 3800,
            "total_assets": 380000,
            "total_liabilities": 295000,
            "total_equity": 85000
          },
          {
            "fiscal_year": 2026,
            "revenue": 440000,
            "ebitda": 155000,
            "total_debt": 95000,
            "cash_equivalents": 40000,
            "interest_expense": 3500,
            "total_assets": 400000,
            "total_liabilities": 300000,
            "total_equity": 100000
          }
        ]
      },
      "MSFT": {
        "company_name": "Microsoft Corporation",
        "sector": "Technology",
        "history": [
          {
            "fiscal_year": 2021,
            "revenue": 168088,
            "ebitda": 80816,
            "total_debt": 58120,
            "cash_equivalents": 14224,
            "interest_expense": 2346,
            "total_assets": 333779,
            "total_liabilities": 191791,
            "total_equity": 141988
          },
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
            "revenue": 245122,
            "ebitda": 125000,
            "total_debt": 45000,
            "cash_equivalents": 40000,
            "interest_expense": 2500,
            "total_assets": 450000,
            "total_liabilities": 220000,
            "total_equity": 230000
          },
          {
            "fiscal_year": 2025,
            "revenue": 285000,
            "ebitda": 145000,
            "total_debt": 42000,
            "cash_equivalents": 55000,
            "interest_expense": 2400,
            "total_assets": 500000,
            "total_liabilities": 230000,
            "total_equity": 270000
          },
          {
            "fiscal_year": 2026,
            "revenue": 330000,
            "ebitda": 170000,
            "total_debt": 40000,
            "cash_equivalents": 70000,
            "interest_expense": 2200,
            "total_assets": 560000,
            "total_liabilities": 240000,
            "total_equity": 320000
          }
        ]
      },
      "GOOGL": {
        "company_name": "Alphabet Inc.",
        "sector": "Technology",
        "history": [
          {
            "fiscal_year": 2021,
            "revenue": 257637,
            "ebitda": 91155,
            "total_debt": 14817,
            "cash_equivalents": 20945,
            "interest_expense": 346,
            "total_assets": 359268,
            "total_liabilities": 107633,
            "total_equity": 251635
          },
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
            "revenue": 340000,
            "ebitda": 105000,
            "total_debt": 13000,
            "cash_equivalents": 28000,
            "interest_expense": 350,
            "total_assets": 430000,
            "total_liabilities": 130000,
            "total_equity": 300000
          },
          {
            "fiscal_year": 2025,
            "revenue": 375000,
            "ebitda": 120000,
            "total_debt": 12000,
            "cash_equivalents": 35000,
            "interest_expense": 300,
            "total_assets": 470000,
            "total_liabilities": 140000,
            "total_equity": 330000
          },
          {
            "fiscal_year": 2026,
            "revenue": 415000,
            "ebitda": 138000,
            "total_debt": 11000,
            "cash_equivalents": 45000,
            "interest_expense": 250,
            "total_assets": 520000,
            "total_liabilities": 150000,
            "total_equity": 370000
          }
        ]
      },
      "AMZN": {
        "company_name": "Amazon.com, Inc.",
        "sector": "Consumer",
        "history": [
          {
            "fiscal_year": 2021,
            "revenue": 469822,
            "ebitda": 59175,
            "total_debt": 48744,
            "cash_equivalents": 36220,
            "interest_expense": 1809,
            "total_assets": 420549,
            "total_liabilities": 282304,
            "total_equity": 138245
          },
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
            "revenue": 635000,
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
            "revenue": 700000,
            "ebitda": 125000,
            "total_debt": 50000,
            "cash_equivalents": 100000,
            "interest_expense": 2800,
            "total_assets": 650000,
            "total_liabilities": 360000,
            "total_equity": 290000
          },
          {
            "fiscal_year": 2026,
            "revenue": 775000,
            "ebitda": 145000,
            "total_debt": 45000,
            "cash_equivalents": 120000,
            "interest_expense": 2500,
            "total_assets": 730000,
            "total_liabilities": 380000,
            "total_equity": 350000
          }
        ]
      },
      "NVDA": {
        "company_name": "NVIDIA Corporation",
        "sector": "Technology",
        "history": [
          {
            "fiscal_year": 2021,
            "revenue": 16675,
            "ebitda": 4532,
            "total_debt": 6965,
            "cash_equivalents": 11561,
            "interest_expense": 184,
            "total_assets": 28791,
            "total_liabilities": 11898,
            "total_equity": 16893
          },
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
            "revenue": 110000,
            "ebitda": 70000,
            "total_debt": 8000,
            "cash_equivalents": 45000,
            "interest_expense": 250,
            "total_assets": 120000,
            "total_liabilities": 30000,
            "total_equity": 90000
          },
          {
            "fiscal_year": 2026,
            "revenue": 150000,
            "ebitda": 95000,
            "total_debt": 7500,
            "cash_equivalents": 70000,
            "interest_expense": 200,
            "total_assets": 180000,
            "total_liabilities": 35000,
            "total_equity": 145000
          }
        ]
      },
      "TSLA": {
        "company_name": "Tesla, Inc.",
        "sector": "Automotive",
        "history": [
          {
            "fiscal_year": 2021,
            "revenue": 53823,
            "ebitda": 9600,
            "total_debt": 6834,
            "cash_equivalents": 17576,
            "interest_expense": 371,
            "total_assets": 62131,
            "total_liabilities": 30548,
            "total_equity": 30189
          },
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
            "ebitda": 16000,
            "total_debt": 5000,
            "cash_equivalents": 32000,
            "interest_expense": 150,
            "total_assets": 125000,
            "total_liabilities": 50000,
            "total_equity": 75000
          },
          {
            "fiscal_year": 2025,
            "revenue": 135000,
            "ebitda": 22000,
            "total_debt": 4500,
            "cash_equivalents": 40000,
            "interest_expense": 140,
            "total_assets": 150000,
            "total_liabilities": 55000,
            "total_equity": 95000
          },
          {
            "fiscal_year": 2026,
            "revenue": 165000,
            "ebitda": 28000,
            "total_debt": 4000,
            "cash_equivalents": 55000,
            "interest_expense": 130,
            "total_assets": 180000,
            "total_liabilities": 60000,
            "total_equity": 120000
          }
        ]
      },
      "META": {
        "company_name": "Meta Platforms, Inc.",
        "sector": "Technology",
        "history": [
          {
            "fiscal_year": 2021,
            "revenue": 117929,
            "ebitda": 54720,
            "total_debt": 13876,
            "cash_equivalents": 16601,
            "interest_expense": 0,
            "total_assets": 165987,
            "total_liabilities": 41108,
            "total_equity": 124879
          },
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
            "cash_equivalents": 55000,
            "interest_expense": 350,
            "total_assets": 260000,
            "total_liabilities": 80000,
            "total_equity": 180000
          },
          {
            "fiscal_year": 2025,
            "revenue": 180000,
            "ebitda": 90000,
            "total_debt": 30000,
            "cash_equivalents": 70000,
            "interest_expense": 300,
            "total_assets": 300000,
            "total_liabilities": 85000,
            "total_equity": 215000
          },
          {
            "fiscal_year": 2026,
            "revenue": 210000,
            "ebitda": 105000,
            "total_debt": 25000,
            "cash_equivalents": 90000,
            "interest_expense": 250,
            "total_assets": 350000,
            "total_liabilities": 90000,
            "total_equity": 260000
          }
        ]
      }
    }
  }
};