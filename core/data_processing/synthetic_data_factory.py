import time
import random
import datetime

class DataFactory:
    @staticmethod
    def generate_deep_dive(ticker="AAPL", scenario="neutral"):
        """
        Generates a synthetic v23_knowledge_graph report.
        Allows for dynamic modification based on scenario.
        """
        base_valuation = 245.50 if ticker == "AAPL" else 150.00

        # Adjust numbers based on scenario
        if scenario == "bear":
            conviction = 3
            recommendation = "Sell"
            modifier = 0.8
        elif scenario == "bull":
            conviction = 9
            recommendation = "Buy"
            modifier = 1.2
        else:
            conviction = 5
            recommendation = "Hold"
            modifier = 1.0

        return {
            "title": f"{ticker} Deep Dive Analysis ({scenario.title()} Case)",
            "file_path": f"synthetic/{ticker}_{scenario}.json",
            "type": "synthetic_report",
            "v23_knowledge_graph": {
                "meta": {
                    "target": ticker,
                    "generated_at": datetime.datetime.now().isoformat(),
                    "model_version": "Adam-v23.5-Synthetic"
                },
                "nodes": {
                    "entity_ecosystem": {
                        "legal_entity": {
                            "name": f"{ticker} Inc.",
                            "jurisdiction": "California, USA"
                        },
                        "management_assessment": {
                            "capital_allocation_score": 9.2 if scenario == "bull" else 6.5,
                            "alignment_analysis": "High ownership structure. Buybacks active.",
                            "key_person_risk": "Low"
                        },
                        "competitive_positioning": {
                            "moat_status": "Wide" if scenario != "bear" else "Narrowing",
                            "technology_risk_vector": "Generative AI integration neutralizes disruption risk."
                        }
                    },
                    "equity_analysis": {
                        "fundamentals": {
                            "revenue_cagr_3yr": "8.5%" if scenario == "bull" else "4.2%",
                            "ebitda_margin_trend": "Expanding" if scenario == "bull" else "Contracting"
                        },
                        "valuation_engine": {
                            "dcf_model": {
                                "wacc": 0.085,
                                "terminal_growth": 0.03,
                                "intrinsic_value": round(3200000.0 * modifier, 2),
                                "intrinsic_share_price": round(base_valuation * modifier, 2)
                            },
                            "multiples_analysis": {
                                "current_ev_ebitda": 22.5,
                                "peer_median_ev_ebitda": 25.0
                            },
                            "price_targets": {
                                "bear_case": round(base_valuation * 0.7, 2),
                                "base_case": round(base_valuation, 2),
                                "bull_case": round(base_valuation * 1.3, 2)
                            }
                        }
                    },
                    "credit_analysis": {
                        "snc_rating_model": {
                            "overall_borrower_rating": "Pass",
                            "facilities": [
                                {"id": "Revolver", "amount": "$5B", "regulatory_rating": "Pass", "collateral_coverage": "Unsecured", "covenant_headroom": ">50%"}
                            ]
                        },
                        "cds_market_implied_rating": "AA",
                        "covenant_risk_analysis": {
                            "primary_constraint": "None",
                            "current_level": 0.0,
                            "breach_threshold": 0.0,
                            "risk_assessment": "Minimal"
                        }
                    },
                    "simulation_engine": {
                        "monte_carlo_default_prob": 0.0001,
                        "quantum_scenarios": [
                            {"name": "Supply Chain Decoupling", "probability": 0.15, "estimated_impact_ev": "-12%"},
                            {"name": "Antitrust Breakup", "probability": 0.05, "estimated_impact_ev": "-25%"}
                        ],
                        "trading_dynamics": {
                            "short_interest": "0.8%",
                            "liquidity_risk": "None"
                        }
                    },
                    "strategic_synthesis": {
                        "m_and_a_posture": "Buyer",
                        "final_verdict": {
                            "recommendation": recommendation,
                            "conviction_level": conviction,
                            "time_horizon": "Long Term",
                            "rationale_summary": f"Automated scenario generation for {scenario} market conditions.",
                            "justification_trace": ["Strong Cash Flow", "Wide Moat", "Valuation Analysis"]
                        }
                    }
                }
            }
        }
