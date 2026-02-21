/**
 * ADAM v23.5 SOVEREIGN MODULES
 * -----------------------------------------------------------------------------
 * Additive extensions for the Sovereign Dashboard.
 * Includes System 2 Debate and Macro Risk Overlay.
 * -----------------------------------------------------------------------------
 */

class SovereignModules {
    constructor() {
        this.init();
    }

    init() {
        console.log("[SovereignModules] Initializing...");
        this.renderDebatePanel();
        this.renderMacroOverlay();

        // Listen for ticker changes if possible (polling for class change)
        this.lastTicker = null;
        setInterval(() => this.checkTickerChange(), 1000);
    }

    checkTickerChange() {
        const activeItem = document.querySelector('.ticker-item.active span');
        if (activeItem) {
            const currentTicker = activeItem.innerText;
            if (currentTicker !== this.lastTicker) {
                this.lastTicker = currentTicker;
                this.simulateDebate(currentTicker);
            }
        }
    }

    renderDebatePanel() {
        if (document.getElementById('sovereign-debate-panel')) return;

        const container = document.querySelector('.container');
        if (!container) return;

        // Create Panel HTML
        const panel = document.createElement('div');
        panel.id = 'sovereign-debate-panel';
        panel.className = 'panel';
        panel.style.cssText = `
            grid-column: span 4;
            height: 250px;
            margin-top: 0;
            display: flex;
            flex-direction: column;
            border: 1px solid #333;
            background: rgba(10, 15, 20, 0.8);
        `;

        panel.innerHTML = `
            <div class="panel-header" style="background: rgba(255, 0, 255, 0.1); border-bottom: 1px solid #555; display:flex; justify-content:space-between; align-items:center; padding-right:10px;">
                <h3 style="color: #ff00ff; margin:0;"><i class="fas fa-comments"></i> SYSTEM 2 DEBATE: BULL vs BEAR</h3>
                <div style="font-size:0.7em; font-family:'JetBrains Mono';">
                    <span style="color:#aaa;">GEOPOLITICAL RISK:</span>
                    <span id="geo-risk-badge" style="color:#ff0055; font-weight:bold;">CALCULATING...</span>
                </div>
            </div>
            <div class="panel-content" style="display: flex; gap: 20px; overflow: hidden; height: 100%;">
                <div style="flex: 1; border-right: 1px dashed #333; padding-right: 10px;">
                    <h4 style="color: #00ff9d; font-size: 0.8em; margin-bottom: 10px;">BULL AGENT (OPTIMIST)</h4>
                    <div id="bull-chat" style="font-family: 'JetBrains Mono'; font-size: 0.75em; color: #aaa; height: 160px; overflow-y: auto;"></div>
                </div>
                <div style="flex: 1;">
                    <h4 style="color: #ff0055; font-size: 0.8em; margin-bottom: 10px;">BEAR AGENT (SKEPTIC)</h4>
                    <div id="bear-chat" style="font-family: 'JetBrains Mono'; font-size: 0.75em; color: #aaa; height: 160px; overflow-y: auto;"></div>
                </div>
            </div>
        `;

        // Insert at bottom of grid
        container.appendChild(panel);
    }

    simulateDebate(ticker = "DEFAULT") {
        const bullDiv = document.getElementById('bull-chat');
        const bearDiv = document.getElementById('bear-chat');
        if(!bullDiv || !bearDiv) return;

        // Clear previous chat
        bullDiv.innerHTML = '';
        bearDiv.innerHTML = '';

        // Update Geo Risk Badge based on ticker
        const riskBadge = document.getElementById('geo-risk-badge');
        if (riskBadge) {
            const riskMap = {
                "AAPL": "HIGH (China)", "TSLA": "MED (China)", "NVDA": "CRITICAL (Export)",
                "MSFT": "LOW", "AMZN": "LOW", "GOOGL": "MED (EU Regs)", "META": "MED (EU Regs)"
            };
            const risk = riskMap[ticker] || "LOW";
            riskBadge.innerText = risk;
            riskBadge.style.color = risk.includes("HIGH") || risk.includes("CRITICAL") ? "#ff0000" : (risk.includes("MED") ? "#f59e0b" : "#00ff9d");
        }

        const INSIGHTS = {
            "AMZN": {
                bull: [
                    "AWS re-acceleration is confirmed by backlog growth.",
                    "Retail margins are expanding due to regional fulfillment model.",
                    "Advertising business is a $50B high-margin juggernaut.",
                    "Prime Video ads provide new free cash flow stream.",
                    "Kuiper satellite internet adds long-term optionality."
                ],
                bear: [
                    "Cloud competition from Azure is eroding market share.",
                    "FTC antitrust lawsuit threatens to break up the company.",
                    "Consumer spending slowdown impacts e-commerce volume.",
                    "GenAI capex is skyrocketing without clear ROI yet.",
                    "International retail segment remains structurally low margin."
                ]
            },
            "GOOGL": {
                bull: [
                    "Search dominance remains intact despite GenAI fears.",
                    "YouTube Shorts monetization is ramping up effectively.",
                    "Gemini Ultra is closing the gap with GPT-4.",
                    "Waymo is the only scalable autonomous driving leader.",
                    "Valuation is the most attractive of the Mag 7."
                ],
                bear: [
                    "Perplexity and ChatGPT are stealing navigational search queries.",
                    "DOJ trial outcome could force divestiture of Ad Manager.",
                    "Search Generative Experience (SGE) cannibalizes ad inventory.",
                    "Cloud unit is a distant third behind AWS and Azure.",
                    "Cultural inertia is slowing down AI product velocity."
                ]
            },
            "META": {
                bull: [
                    "Llama 3 is the standard for open-source AI, creating ecosystem lock-in.",
                    "Reels monetization has reached parity with Feed.",
                    "Advantage+ AI tools are driving massive ROAS for advertisers.",
                    "Cost discipline 'Year of Efficiency' has permanently improved margins.",
                    "WhatsApp monetization is finally unlocking."
                ],
                bear: [
                    "Reality Labs burn rate of $15B/year is value destructive.",
                    "TikTok remains a formidable competitor for time share.",
                    "Regulatory scrutiny on child safety is intensifying globally.",
                    "Apple's privacy changes have permanently impaired targeting signal.",
                    "Mark Zuckerberg's voting control ignores shareholder wishes."
                ]
            },
            "AAPL": {
                bull: [
                    "Services revenue is compounding at 14% annually.",
                    "iPhone upgrade supercycle driven by AI features is imminent.",
                    "Cash pile of $160B provides infinite optionality.",
                    "Ecosystem lock-in is at an all-time high.",
                    "Buybacks are effectively a floor on the stock price."
                ],
                bear: [
                    "China regulatory headwinds are structural, not cyclical.",
                    "Hardware margins are compressing due to component costs.",
                    "Vision Pro adoption is slower than the Apple Watch curve.",
                    "Antitrust litigation in the EU threatens the App Store moat.",
                    "Valuation at 30x PE leaves no room for error."
                ]
            },
            "MSFT": {
                bull: [
                    "Azure is gaining share in the AI compute layer.",
                    "Copilot monetization is faster than SaaS adoption.",
                    "Enterprise moat is unassailable.",
                    "Gaming division (Activision) adds a new growth vector.",
                    "Management execution is flawless."
                ],
                bear: [
                    "AI capex spending is diluting free cash flow margins.",
                    "PC market saturation limits Windows growth.",
                    "Cloud optimization is still a headwind for Azure.",
                    "Regulatory scrutiny on AI partnerships is increasing.",
                    "Valuation is priced for perfection."
                ]
            },
            "TSLA": {
                bull: [
                    "FSD solve is closer than the market thinks.",
                    "Energy storage business is growing triple digits.",
                    "Robotaxi unit economics are transformative.",
                    "Cybertruck production ramp is proving the skeptics wrong.",
                    "Cost of goods sold per vehicle is lowest in industry."
                ],
                bear: [
                    "Auto margins have collapsed due to price cuts.",
                    "EV demand is slowing globally.",
                    "Competition from BYD is eroding market share.",
                    "Key man risk is elevated.",
                    "FSD regulatory approval is years away."
                ]
            },
            "NVDA": {
                bull: [
                    "Demand for H100/Blackwell exceeds supply for 18 months.",
                    "CUDA software moat is impenetrable.",
                    "Sovereign AI is a new multi-billion dollar vertical.",
                    "Data center transformation is a $1T opportunity.",
                    "Margins are software-like (75% gross)."
                ],
                bear: [
                    "Customer concentration risk (Hyperscalers) is extreme.",
                    "Competition from custom silicon (TPU, Trainium) is rising.",
                    "China export restrictions cap total addressable market.",
                    " cyclical correction in semi demand is inevitable.",
                    "Valuation assumes perpetual exponential growth."
                ]
            },
            "DEFAULT": {
                bull: [
                    "Revenue growth remains robust despite headwinds.",
                    "Cash flow conversion is improving, creating a buffer.",
                    "Market dominance justifies the premium valuation.",
                    "Strategic reserves are sufficient to cover liabilities.",
                    "Innovation pipeline suggests upside potential."
                ],
                bear: [
                    "Leverage is creeping up; Debt/EBITDA is elevated.",
                    "Adjustments to EBITDA are masking margin compression.",
                    "Regulatory risks are being underpriced.",
                    "Refinancing wall is approaching in a high-rate environment.",
                    "Consumer demand softening is a leading indicator."
                ]
            }
        };

        const data = INSIGHTS[ticker] || INSIGHTS["DEFAULT"];
        const bullMsgs = data.bull;
        const bearMsgs = data.bear;

        let i = 0;
        // Clear any existing interval
        if (this.chatInterval) clearInterval(this.chatInterval);

        const addMsg = () => {
            if (i >= bullMsgs.length) {
                clearInterval(this.chatInterval);
                return;
            }

            // Bull speaks
            const bMsg = document.createElement('div');
            bMsg.style.marginBottom = "8px";
            bMsg.style.borderLeft = "2px solid #00ff9d";
            bMsg.style.paddingLeft = "8px";
            bMsg.innerText = `> ${bullMsgs[i]}`;
            bullDiv.appendChild(bMsg);
            bullDiv.scrollTop = bullDiv.scrollHeight;

            // Bear responds after delay
            setTimeout(() => {
                const rMsg = document.createElement('div');
                rMsg.style.marginBottom = "8px";
                rMsg.style.borderLeft = "2px solid #ff0055";
                rMsg.style.paddingLeft = "8px";
                rMsg.innerText = `> ${bearMsgs[i]}`;
                bearDiv.appendChild(rMsg);
                bearDiv.scrollTop = bearDiv.scrollHeight;
                i++;
            }, 1500);
        };

        this.chatInterval = setInterval(addMsg, 4000);
        addMsg();
    }

    renderMacroOverlay() {
        // Find the quantitative chart container
        const chartContainer = document.querySelector('#quantContent .chart-container');
        if (!chartContainer) return;

        // Ensure relative positioning
        chartContainer.style.position = 'relative';

        // Check if canvas exists
        if (document.getElementById('macro-overlay')) return;

        // Create Canvas Overlay
        const canvas = document.createElement('canvas');
        canvas.id = 'macro-overlay';
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.pointerEvents = 'none'; // Click through
        canvas.style.zIndex = '10';

        chartContainer.appendChild(canvas);

        // Adjust resolution
        const rect = chartContainer.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;

        const ctx = canvas.getContext('2d');

        // Mouse interaction
        let mouseX = -1000;
        let mouseY = -1000;
        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;
            mouseY = e.clientY - rect.top;
        });

        // Draw Animation Loop
        const particles = [];
        for(let j=0; j<20; j++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                r: Math.random() * 3 + 1,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5
            });
        }

        const animate = () => {
            if (!document.getElementById('macro-overlay')) return; // Stop if removed

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw Warning Label
            ctx.font = "10px JetBrains Mono";
            ctx.fillStyle = "rgba(255, 0, 0, 0.5)";
            ctx.fillText("MACRO CONTAGION MATRIX: ACTIVE", 10, 20);

            // Update & Draw Particles
            ctx.fillStyle = "rgba(255, 0, 0, 0.2)";
            particles.forEach(p => {
                p.x += p.vx;
                p.y += p.vy;

                // Mouse Repel/Attract
                const dx = p.x - mouseX;
                const dy = p.y - mouseY;
                const dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < 100) {
                    const angle = Math.atan2(dy, dx);
                    const force = (100 - dist) / 100;
                    // Repel
                    p.vx += Math.cos(angle) * force * 0.5;
                    p.vy += Math.sin(angle) * force * 0.5;
                }

                // Bounce
                if(p.x < 0 || p.x > canvas.width) p.vx *= -1;
                if(p.y < 0 || p.y > canvas.height) p.vy *= -1;

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fill();

                // Connect lines if close
                particles.forEach(p2 => {
                    const dx = p.x - p2.x;
                    const dy = p.y - p2.y;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    if(dist < 50) {
                        ctx.beginPath();
                        ctx.strokeStyle = `rgba(255, 0, 0, ${0.2 - dist/250})`;
                        ctx.lineWidth = 0.5;
                        ctx.moveTo(p.x, p.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.stroke();
                    }
                });
            });

            requestAnimationFrame(animate);
        };
        animate();
    }
}

// Auto-Launch
document.addEventListener('DOMContentLoaded', () => {
    // Wait slightly for main DOM to settle
    setTimeout(() => {
        new SovereignModules();
    }, 1000);
});
