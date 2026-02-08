
// Agent Portfolio Loader
// Injects recent work items into the Agent Details Modal
(function() {
    let portfolioData = null;

    async function loadPortfolioData() {
        if (portfolioData) return;
        try {
            const response = await fetch('data/agent_portfolio.json');
            portfolioData = await response.json();
            console.log("Agent Portfolio Loaded", Object.keys(portfolioData).length, "agents");
        } catch (e) {
            console.warn("Failed to load agent portfolio", e);
            portfolioData = {};
        }
    }

    // Observer to detect when the modal is opened/populated
    const observer = new MutationObserver(async (mutations) => {
        for (const mutation of mutations) {
            // Check if modal title changed (meaning new agent selected)
            if (mutation.target.id === 'modal-title' || mutation.target.parentNode.id === 'modal-title') {
                await injectPortfolio();
            }
        }
    });

    // Wait for DOM
    document.addEventListener('DOMContentLoaded', () => {
        loadPortfolioData();

        // Find the modal title element
        const titleEl = document.getElementById('modal-title');
        if (titleEl) {
            observer.observe(titleEl, { childList: true, subtree: true, characterData: true });
        } else {
            console.warn("Agent Modal Title element not found. Portfolio injection disabled.");
        }
    });

    async function injectPortfolio() {
        if (!portfolioData) await loadPortfolioData();

        const agentName = document.getElementById('modal-title').innerText.trim();
        const works = portfolioData[agentName] || [];

        // Target injection point - we want to insert before the "Overview" content
        // In the existing `agents.html`, the content is populated into `details-content`.
        // We can prepend to `details-content` or check if we already injected.

        const detailsContainer = document.getElementById('details-content');
        if (!detailsContainer) return;

        // Remove existing portfolio if any
        const existing = document.getElementById('agent-portfolio-section');
        if (existing) existing.remove();

        if (works.length === 0) return;

        // Create Portfolio Section
        const div = document.createElement('div');
        div.id = 'agent-portfolio-section';
        div.className = 'glass-panel';
        div.style.padding = '15px';
        div.style.marginBottom = '20px';
        div.style.border = '1px solid var(--primary-color)';
        div.style.background = 'rgba(0, 243, 255, 0.05)';

        div.innerHTML = `
            <h4 style="color: var(--primary-color); margin-top: 0; display:flex; justify-content:space-between; align-items:center;">
                <span class="mono">INTELLIGENCE OUTPUT</span>
                <span style="font-size:0.7rem; background:rgba(0, 243, 255, 0.2); padding:2px 6px; border-radius:4px;">${works.length} FILES</span>
            </h4>
            <div style="max-height: 150px; overflow-y: auto; margin-top:10px;" class="custom-scrollbar">
                ${works.map(w => `
                    <a href="${w.filename}" target="_blank" style="display:block; padding:8px; border-bottom:1px solid rgba(255,255,255,0.1); text-decoration:none; color:#ddd; transition:all 0.2s;" onmouseover="this.style.background='rgba(0,243,255,0.1)'" onmouseout="this.style.background='transparent'">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:bold; font-size:0.85rem;">${w.title}</span>
                            <span style="font-size:0.7rem; color:#888; font-family:'JetBrains Mono'">${w.date}</span>
                        </div>
                        <div style="font-size:0.7rem; color:var(--primary-color); margin-top:2px;">${w.type}</div>
                    </a>
                `).join('')}
            </div>
        `;

        // Inject at the top of the details container
        detailsContainer.insertBefore(div, detailsContainer.firstChild);
    }

})();
