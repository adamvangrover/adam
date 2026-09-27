import json
from scripts.adversarial_pd_challenger import calculate_forward_looking_pd
import random

def generate_report():
    # Universe of private markets, LBOs, AI ecosystems, targeting distressed/high-yield (B- to CCC+)
    companies = [
        {
            'name': 'Anthropic Infrastructure SPV', 'ticker': 'PRIV-AI-1',
            'current_debt': 1500.0, 'short_term_debt': 800.0, 'projected_fcf': -200.0,
            'implied_volatility': 0.85, 'forward_interest_rate': 0.08, 'equity_value': 4000.0, 'macro_stress_factor': 0.60,
            'rumor': 'Seeking bridge facility amid GPU supply chain delays. High cash burn.'
        },
        {
            'name': 'CloudScale Software (LBO)', 'ticker': 'CS-LBO',
            'current_debt': 3200.0, 'short_term_debt': 500.0, 'projected_fcf': 150.0,
            'implied_volatility': 0.65, 'forward_interest_rate': 0.11, 'equity_value': 800.0, 'macro_stress_factor': 0.70,
            'rumor': 'Covenant breach likely in Q3. Sponsor in talks for PIK toggle.'
        },
        {
            'name': 'NeuralNet Edge (M&A Target)', 'ticker': 'NN-MA',
            'current_debt': 400.0, 'short_term_debt': 350.0, 'projected_fcf': 10.0,
            'implied_volatility': 0.95, 'forward_interest_rate': 0.09, 'equity_value': 250.0, 'macro_stress_factor': 0.50,
            'rumor': 'Acquisition by Big Tech stalled by FTC. Liquidity crunch imminent.'
        },
        {
            'name': 'SaaS Rollup Holdings', 'ticker': 'SR-HLD',
            'current_debt': 1800.0, 'short_term_debt': 400.0, 'projected_fcf': 80.0,
            'implied_volatility': 0.55, 'forward_interest_rate': 0.12, 'equity_value': 600.0, 'macro_stress_factor': 0.75,
            'rumor': 'Integration of latest acquisition failing. Margins compressing rapidly.'
        },
        {
            'name': 'CLO Tranche X (Tech Mezzanine)', 'ticker': 'CLO-X-MEZZ',
            'current_debt': 500.0, 'short_term_debt': 100.0, 'projected_fcf': 45.0,
            'implied_volatility': 0.75, 'forward_interest_rate': 0.14, 'equity_value': 150.0, 'macro_stress_factor': 0.65,
            'rumor': 'Underlying loan defaults creeping up. Downgrade watch.'
        }
    ]

    results = []
    for c in companies:
        res = calculate_forward_looking_pd(
            name=c['name'],
            ticker=c['ticker'],
            current_debt=c['current_debt'],
            short_term_debt=c['short_term_debt'],
            projected_fcf=c['projected_fcf'],
            implied_volatility=c['implied_volatility'],
            forward_interest_rate=c['forward_interest_rate'],
            equity_value=c['equity_value'],
            macro_stress_factor=c['macro_stress_factor']
        )

        pd = res['adversarial_pd_challenger']

        # Simulate ratings based on PD mapping
        rating = 'B+' if pd < 0.10 else 'B' if pd < 0.15 else 'B-' if pd < 0.20 else 'CCC+' if pd < 0.25 else 'CCC' if pd < 0.35 else 'CCC-' if pd < 0.5 else 'D'

        # Evaluate confidence based on volatility and stress factors
        confidence_score = round(max(0.4, 1.0 - (c['implied_volatility'] * 0.5) - ((1.0 - c['macro_stress_factor']) * 0.3)), 2)

        results.append({
            'entity': c['name'],
            'sector': 'Private Credit / High Yield Tech',
            'market_rumors': c['rumor'],
            'pd_challenger': round(pd, 4),
            'implied_rating': rating,
            'confidence_score': confidence_score,
            'model_internals': {
                'v_firm_stressed': round(res['inputs']['v_firm_stressed'], 2),
                'liquidity_pd': round(res['components']['liquidity_pd'], 4),
                'structural_pd': round(res['components']['structural_pd'], 4)
            }
        })

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    generate_report()
