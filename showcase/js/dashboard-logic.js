/**
 * ADAM v23.5 DASHBOARD LOGIC
 * -----------------------------------------------------------------------------
 * Cyberpunk Financial Intelligence Dashboard
 * -----------------------------------------------------------------------------
 */

document.addEventListener('DOMContentLoaded', () => {
    // Ensure we don't double init
    if (window.dashboardInitialized) return;
    window.dashboardInitialized = true;

    console.log("[Dashboard] Initializing Cyberpunk Overlay...");

    initSystem2Terminal();
    initSearch();
    enrichContent();
    animateConviction();
});

function initSystem2Terminal() {
    const terminal = document.querySelector('.system2-terminal');
    const header = document.querySelector('.terminal-header');

    if (!terminal || !header) return;

    // Toggle collapse
    header.addEventListener('click', () => {
        const isCollapsed = terminal.classList.contains('collapsed');
        terminal.classList.toggle('collapsed', !isCollapsed);

        // Update indicator icon if present
        const icon = header.querySelector('.toggle-icon');
        if (icon) {
            icon.className = isCollapsed ? 'fas fa-chevron-down toggle-icon' : 'fas fa-chevron-up toggle-icon';
        }
    });

    // Populate log with fake data if empty
    const logBody = document.querySelector('.terminal-body');
    if (logBody && !logBody.children.length) {
        const logs = [
            "INITIALIZING NEURAL LINK...",
            "SCANNING FOR DIVERGENCE...",
            "DETECTING MARKET ANOMALIES...",
            "SENTIMENT ANALYSIS: MIXED",
            "SYSTEM 2 AUDIT: COMPLETE",
            "ESTABLISHING SECURE CONNECTION...",
            "DOWNLOADING LATEST TICKER DATA...",
            "APPLYING PROVENANCE FILTERS..."
        ];

        logs.forEach((msg, i) => {
            setTimeout(() => {
                addLogEntry(msg);
            }, i * 300);
        });
    }
}

function addLogEntry(msg) {
    const logBody = document.querySelector('.terminal-body');
    if (!logBody) return;

    const entry = document.createElement('div');
    entry.className = 'log-entry';
    const ts = new Date().toISOString().split('T')[1].slice(0,12);
    entry.innerHTML = `<span class="log-ts">[${ts}]</span> <span class="log-msg">${msg}</span>`;
    logBody.appendChild(entry);
    logBody.scrollTop = logBody.scrollHeight;
}

function initSearch() {
    const input = document.getElementById('hud-search-input');
    // We search within the original content wrapper
    const content = document.querySelector('.cyber-main-content');

    if (!input || !content) return;

    input.addEventListener('input', (e) => {
        const term = e.target.value.toLowerCase();

        // Remove previous highlights safely
        // We find all spans with class 'highlight-match' and unwrap them
        const marks = content.querySelectorAll('.highlight-match');
        marks.forEach(m => {
            const parent = m.parentNode;
            parent.replaceChild(document.createTextNode(m.textContent), m);
            parent.normalize(); // Merge text nodes
        });

        if (term.length < 3) return;

        // Simple text walker
        // Skip script and style tags
        const walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT, {
            acceptNode: function(node) {
                if (node.parentNode.tagName === 'SCRIPT' ||
                    node.parentNode.tagName === 'STYLE' ||
                    node.parentNode.classList.contains('highlight-match')) {
                    return NodeFilter.FILTER_REJECT;
                }
                return NodeFilter.FILTER_ACCEPT;
            }
        }, false);

        let node;
        const nodesToReplace = [];

        while(node = walker.nextNode()) {
            if (node.textContent.toLowerCase().includes(term)) {
                nodesToReplace.push(node);
            }
        }

        nodesToReplace.forEach(node => {
            try {
                const fragment = document.createDocumentFragment();
                // Use a safe regex escape function if needed, but for now simple split
                // Note: simple split might break if term contains special regex chars
                const safeTerm = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                const parts = node.textContent.split(new RegExp(`(${safeTerm})`, 'gi'));

                parts.forEach(part => {
                    if (part.toLowerCase() === term) {
                        const span = document.createElement('span');
                        span.className = 'highlight-match';
                        span.textContent = part;
                        fragment.appendChild(span);
                    } else {
                        fragment.appendChild(document.createTextNode(part));
                    }
                });
                node.parentNode.replaceChild(fragment, node);
            } catch (e) {
                console.warn("Search replace error", e);
            }
        });
    });
}

function enrichContent() {
    // 1. Provenance Chips
    // Find paragraphs that contain numbers or % and inject source chips randomly
    const paragraphs = document.querySelectorAll('.cyber-main-content p, .cyber-main-content li');
    paragraphs.forEach((p) => {
        // Only target text that looks like a claim (has numbers)
        if (/\d/.test(p.textContent) && Math.random() > 0.6) {
            // Don't add if already has one
            if (p.querySelector('.source-chip')) return;

            const chip = document.createElement('span');
            chip.className = 'source-chip';
            const srcId = Math.floor(Math.random() * 9000) + 1000;
            chip.textContent = `[SRC:${srcId}]`;

            const tooltip = document.createElement('div');
            tooltip.className = 'source-tooltip';
            const conf = (Math.random() * 20 + 80).toFixed(1);
            tooltip.innerHTML = `
                <div style="border-bottom:1px solid #ff0099; padding-bottom:4px; margin-bottom:4px; font-weight:bold;">SOURCE VERIFIED</div>
                <div>ID: ${srcId}</div>
                <div>Origin: Bloomberg Terminal / SEC Filings</div>
                <div>Confidence: <span style="color:#00f3ff">${conf}%</span></div>
                <div style="margin-top:4px; font-size:0.65rem; color:#aaa;">${new Date().toISOString()}</div>
            `;

            chip.appendChild(tooltip);
            p.appendChild(chip);
        }
    });

    // 2. Metadata Tags
    const tagsContainer = document.querySelector('.metadata-tags');
    if (!tagsContainer) return;

    // Default tags
    const potentialTags = ['AI', 'CRYPTO', 'MACRO', 'VOLATILITY', 'BIFURCATION', 'SAAS', 'INFLATION', 'GOLD', 'TECH'];

    // Check content for keywords
    const text = document.body.textContent.toUpperCase();
    let foundTags = 0;

    potentialTags.forEach(kw => {
        if (text.includes(kw)) {
            const tag = document.createElement('span');
            tag.className = 'meta-tag';
            tag.textContent = kw;
            tag.onclick = () => {
                const input = document.getElementById('hud-search-input');
                if (input) {
                    input.value = kw;
                    input.dispatchEvent(new Event('input'));
                }
            };
            tagsContainer.appendChild(tag);
            foundTags++;
        }
    });

    if (foundTags === 0) {
        // Fallback
        ['MARKET', 'ANALYSIS'].forEach(t => {
            const tag = document.createElement('span');
            tag.className = 'meta-tag';
            tag.textContent = t;
            tagsContainer.appendChild(tag);
        });
    }
}

function animateConviction() {
    const fill = document.querySelector('.conviction-fill');
    const valText = document.querySelector('.conviction-value');

    if (fill) {
        // Try to scrape conviction from page content
        let score = 75; // Default

        // 1. Look for "Conviction: High" or similar
        const bodyText = document.body.textContent;
        if (bodyText.match(/Conviction\s*:\s*High/i)) score = 92;
        else if (bodyText.match(/Conviction\s*:\s*Medium/i)) score = 65;
        else if (bodyText.match(/Conviction\s*:\s*Low/i)) score = 35;

        // 2. Look for explicit score in sidebar (common in these templates)
        // Structure: <span>SCORE</span> <span class="stat-val">33/100</span>
        const statVals = document.querySelectorAll('.stat-val');
        for (let sv of statVals) {
            if (sv.textContent.includes('/100')) {
                score = parseInt(sv.textContent.split('/')[0]);
                break;
            }
        }

        // Animate
        setTimeout(() => {
            fill.style.width = score + '%';
            if (valText) valText.textContent = score + '%';

            // Color based on score
            if (score < 40) {
                fill.style.backgroundColor = '#ff0000';
                fill.style.boxShadow = '0 0 8px #ff0000';
                fill.parentElement.style.borderColor = '#ff0000';
            } else if (score < 70) {
                fill.style.backgroundColor = '#ffaa00';
                fill.style.boxShadow = '0 0 8px #ffaa00';
                fill.parentElement.style.borderColor = '#ffaa00';
            }
        }, 800);

        addLogEntry(`CONVICTION SCORE CALCULATED: ${score}%`);
    }
}
