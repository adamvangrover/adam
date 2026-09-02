import re
import json

with open('scripts/data_generation/live_data.json', 'r') as f:
    live_data = json.load(f)

# Embed the massive dataset (16,000 items)
live_data_js = json.dumps(live_data)

new_init_world = f"""
        const liveData = {live_data_js};
        
        function initWorld(count) {{
            // Use the actual generated massive dataset directly (up to provided count, which will be 16000)
            const actualCount = Math.min(count, liveData.length);
            for (let i = 0; i < actualCount; i++) {{
                const template = liveData[i];
                
                let ebitda = template.baseEbitda;
                let leverageBase = Math.max(1.0, rng.nextGaussian(4.5, 1.2));
                
                // Adjust leverage assumptions based on asset class
                if (template.sector.includes('Crypto')) {{
                    leverageBase = Math.max(0.0, rng.nextGaussian(1.0, 0.5)); 
                }} else if (template.sector.includes('Sovereign')) {{
                    leverageBase = Math.max(5.0, rng.nextGaussian(10.0, 2.0)); 
                }} else if (template.sector.includes('Fiat') || template.sector.includes('Rates')) {{
                    leverageBase = 1.0; 
                }} else if (template.sector.includes('CDS') || template.sector.includes('Structured')) {{
                    leverageBase = Math.max(3.0, rng.nextGaussian(6.0, 1.5)); 
                }}

                const totalDebt = ebitda * leverageBase;
                
                // LGD Tiered Facilities
                const tlbAmount = totalDebt * 0.75;
                const revolverAmount = totalDebt * 0.25;
                
                // Quantum State properties for tail risk based on asset specific volatility
                const quantumDrift = template.volatility;
                
                worldEntities.push({{
                    id: `node_${{Math.floor(rng.nextFloat() * 1000000).toString(16).padStart(5, '0')}}`,
                    name: template.name,
                    ticker: template.ticker,
                    sector: template.sector,
                    baseEbitda: ebitda,
                    baseDebt: totalDebt,
                    quantumDrift: quantumDrift, // Structural tail component
                    facilities: {{
                        tlb: {{ amount: tlbAmount, drawn: tlbAmount, rate: 0.075, lgd: 0.35 }},
                        revolver: {{ amount: revolverAmount, drawn: revolverAmount * 0.1, rate: 0.06, lgd: 0.15 }}
                    }},
                    rating: 'B', 
                    history: []
                }});
            }}
        }}
"""

with open('risk_ops.html', 'r') as f:
    html = f.read()

# Replace initWorld function and liveData array
html = re.sub(r'const liveData =.*?function initWorld\(count\) \{.*?\}(?=\s*function calculateMetrics)', new_init_world, html, flags=re.DOTALL)

# Update the worker init call in the HTML to use 16000 instead of 10000 to match our dataset
html = html.replace("worker.postMessage({ action: 'INIT', payload: { count: 10000 } });", "worker.postMessage({ action: 'INIT', payload: { count: 16000 } });")
html = html.replace("document.getElementById('sys-entity-count').innerText = \"10,000\";", "document.getElementById('sys-entity-count').innerText = \"16,000\";")

with open('risk_ops.html', 'w') as f:
    f.write(html)

print("risk_ops.html updated successfully with massive dataset.")
