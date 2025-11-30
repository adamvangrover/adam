from core.schemas.v23_5_schema import HyperDimensionalKnowledgeGraph, V23KnowledgeGraph
import json

def test_schema_validity():
    # Basic instantiation test
    try:
        # We won't fully instantiate deep objects here, just checking import and model existence
        # But let's try to validate the JSON example from the prompt
        example_json = {
            "v23_knowledge_graph": {
                "meta": {
                    "target": "TEST",
                    "generated_at": "2023-10-27",
                    "model_version": "Adam-v23.5"
                },
                "nodes": {
                    "entity_ecosystem": {
                        "legal_entity": { "name": "Test Corp", "lei": "123", "jurisdiction": "US" },
                        "management_assessment": {
                            "capital_allocation_score": 0.8,
                            "alignment_analysis": "Good",
                            "key_person_risk": "Low"
                        },
                        "competitive_positioning": {
                            "moat_status": "Wide",
                            "technology_risk_vector": "Low"
                        }
                    },
                    "equity_analysis": {
                        "fundamentals": {
                            "revenue_cagr_3yr": "10%",
                            "ebitda_margin_trend": "Expanding"
                        },
                        "valuation_engine": {
                            "dcf_model": {
                                "wacc": 0.08,
                                "terminal_growth": 0.02,
                                "intrinsic_value": 100.0
                            },
                            "multiples_analysis": {
                                "current_ev_ebitda": 10.0,
                                "peer_median_ev_ebitda": 12.0
                            },
                            "price_targets": {
                                "bear_case": 80.0,
                                "base_case": 100.0,
                                "bull_case": 120.0
                            }
                        }
                    },
                    "credit_analysis": {
                        "snc_rating_model": {
                            "overall_borrower_rating": "Pass",
                            "facilities": [
                                {
                                    "id": "Term Loan B",
                                    "amount": "500M",
                                    "regulatory_rating": "Pass",
                                    "collateral_coverage": "2.0x",
                                    "covenant_headroom": "High"
                                }
                            ]
                        },
                        "cds_market_implied_rating": "BBB",
                        "covenant_risk_analysis": {
                            "primary_constraint": "Net Leverage Ratio",
                            "current_level": 3.0,
                            "breach_threshold": 4.0,
                            "risk_assessment": "Low"
                        }
                    },
                    "simulation_engine": {
                        "monte_carlo_default_prob": 0.01,
                        "quantum_scenarios": [
                            { "name": "War", "probability": 0.01, "estimated_impact_ev": "-20%" }
                        ],
                        "trading_dynamics": {
                            "short_interest": "5%",
                            "liquidity_risk": "Low"
                        }
                    },
                    "strategic_synthesis": {
                        "m_and_a_posture": "Neutral",
                        "final_verdict": {
                            "recommendation": "Hold",
                            "conviction_level": 7,
                            "time_horizon": "12-Month",
                            "rationale_summary": "Solid but priced in.",
                            "justification_trace": [
                                "Reason 1: Good moat.",
                                "Reason 2: High valuation."
                            ]
                        }
                    }
                }
            }
        }

        model = HyperDimensionalKnowledgeGraph(**example_json)
        print("Schema validation successful.")

    except Exception as e:
        print(f"Schema validation failed: {e}")
        raise e

if __name__ == "__main__":
    test_schema_validity()
