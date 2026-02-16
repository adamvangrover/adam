window.CyberDashboard = {
    init: function() {
        console.log("Initializing Cyberpunk Dashboard Protocol v2.0...");
        this.injectAssets();
        this.injectLayout();
        this.injectMetadataHeader();
        this.injectSwarmWidget();
        this.injectFocusMode();
        this.processCitations();
        this.injectSystem2();
        this.setupSearch();
        this.ensureDataLoaded();
        console.log("Dashboard Protocol Online.");
    },

    ensureDataLoaded: function() {
        // Attempt to sync with global mock data if available
        if (typeof window.MOCK_DATA !== 'undefined') {
            console.log("SYNC: Connection to Core Data Grid established.");
            this.mockData = window.MOCK_DATA;
            // Re-run tagging now that we have data
            const container = document.getElementById('header-tags');
            if (container) container.innerHTML = ''; // Clear existing
            this.generateTags();
        } else {
            console.warn("SYNC: Core Data Grid offline. Operating in autonomous mode.");
            // Retry once after 2 seconds (in case of async load)
            setTimeout(() => {
                if (typeof window.MOCK_DATA !== 'undefined' && !this.mockData) {
                    this.mockData = window.MOCK_DATA;
                    this.generateTags();
                }
            }, 2000);
        }
    },

    injectAssets: function() {
        const head = document.head;

        // FontAwesome
        if (!document.querySelector('link[href*="font-awesome"]')) {
            const fa = document.createElement('link');
            fa.rel = 'stylesheet';
            fa.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css';
            head.appendChild(fa);
        }

        // Google Fonts
        if (!document.querySelector('link[href*="JetBrains+Mono"]')) {
            const fonts = document.createElement('link');
            fonts.rel = 'stylesheet';
            fonts.href = 'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Courier+Prime:wght@400;700&display=swap';
            head.appendChild(fonts);
        }
    },

    injectLayout: function() {
        if (document.querySelector('.dashboard-container')) return;

        const body = document.body;
        const container = document.createElement('div');
        container.className = 'dashboard-container';

        const header = document.createElement('header');
        header.className = 'metadata-header';
        header.id = 'cyber-header';
        container.appendChild(header);

        const contentArea = document.createElement('main');
        contentArea.className = 'dashboard-content';
        contentArea.id = 'main-content-area';

        let existingContent = document.querySelector('.newsletter-wrapper') ||
                              document.querySelector('.paper-sheet') ||
                              document.querySelector('.content-panel');

        if (existingContent) {
            contentArea.appendChild(existingContent);
        } else {
            while (body.firstChild) {
                if (body.firstChild === container) break;
                contentArea.appendChild(body.firstChild);
            }
        }

        container.appendChild(contentArea);

        const terminal = document.createElement('div');
        terminal.className = 'system2-terminal';
        terminal.id = 'sys2-terminal';
        container.appendChild(terminal);

        document.body.appendChild(container);
    },

    injectSwarmWidget: function() {
        const sidebar = document.querySelector('.sidebar') || document.querySelector('aside');
        if (!sidebar) return;

        const widget = document.createElement('div');
        widget.className = 'swarm-widget';
        widget.innerHTML = `
            <div class="sidebar-title" style="color:var(--neon-green)">SWARM INTELLIGENCE</div>
            <div id="swarm-content"></div>
        `;

        const content = widget.querySelector('#swarm-content');

        // Mock Agents (Fallback or from Data)
        let agents = [
            { name: 'MarketScanner_v9', status: 'active' },
            { name: 'SentimentOracle', status: 'busy' },
            { name: 'RiskGuardian', status: 'active' }
        ];

        if (this.mockData && this.mockData.agents) {
             // If we had live agent data structure
        }

        agents.forEach(agent => {
            const row = document.createElement('div');
            row.className = 'swarm-agent-row';
            row.innerHTML = `
                <div class="agent-status-dot ${agent.status === 'busy' ? 'busy' : ''}"></div>
                <span>${agent.name}</span>
            `;
            content.appendChild(row);
        });

        // Add to top of sidebar
        sidebar.insertBefore(widget, sidebar.firstChild);
    },

    injectFocusMode: function() {
        const header = document.getElementById('cyber-header');
        if (!header) return;

        const btn = document.createElement('button');
        btn.className = 'cyber-btn';
        btn.textContent = 'FOCUS MODE';
        btn.style.marginLeft = '10px';
        btn.onclick = () => {
            document.body.classList.toggle('focus-mode-active');
            btn.classList.toggle('active');
            this.logInteraction(document.body.classList.contains('focus-mode-active') ? "FOCUS MODE ENGAGED" : "FOCUS MODE DISENGAGED");
        };

        header.appendChild(btn);

        // Hover logic
        const content = document.getElementById('main-content-area');
        if (content) {
            content.addEventListener('mouseover', (e) => {
                if (!document.body.classList.contains('focus-mode-active')) return;
                if (e.target.tagName === 'P') {
                    const paragraphs = content.querySelectorAll('p');
                    paragraphs.forEach(p => {
                        if (p !== e.target) p.classList.add('focus-dimmed');
                        else p.classList.remove('focus-dimmed');
                    });
                }
            });

            content.addEventListener('mouseout', () => {
                if (!document.body.classList.contains('focus-mode-active')) return;
                const paragraphs = content.querySelectorAll('p');
                paragraphs.forEach(p => p.classList.remove('focus-dimmed'));
            });
        }
    },

    logInteraction: function(msg) {
        const terminal = document.getElementById('sys2-logs');
        if (!terminal) return;
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        const time = new Date().toLocaleTimeString();
        entry.innerHTML = `<span class="log-timestamp">[${time}]</span> <span style="color:#fff">${msg}</span>`;
        terminal.appendChild(entry);
        terminal.parentElement.scrollTop = terminal.parentElement.scrollHeight;
    },

    injectMetadataHeader: function() {
        const header = document.getElementById('cyber-header');
        if (!header) return;

        const title = document.title.replace('ADAM', '').replace('::', '').trim() || "MARKET MAYHEM";

        // Calculate Conviction
        let conviction = 50;
        const text = document.body.innerText.toUpperCase();
        if (text.includes("HIGH CONVICTION")) conviction = 90;
        else if (text.includes("MEDIUM CONVICTION")) conviction = 60;
        else if (text.includes("LOW CONVICTION")) conviction = 30;

        const scoreMatch = text.match(/CONVICTION[:\s]*(\d+)/);
        if (scoreMatch) conviction = parseInt(scoreMatch[1]);

        header.innerHTML = `
            <div class="header-title">
                <i class="fas fa-biohazard"></i>
                <span class="glitch-text">${title}</span>
            </div>
            <div id="header-tags" style="display:flex; align-items:center;"></div>
            <div class="conviction-gauge-container">
                <div class="conviction-label">SYS.CONVICTION</div>
                <div class="conviction-bar">
                    <div class="conviction-fill" style="width: 0%"></div>
                </div>
                <div class="conviction-label">${conviction}%</div>
            </div>
            <div class="search-container">
                <input type="text" class="cyber-search" id="cyber-search-input" placeholder="SEARCH INTEL...">
            </div>
        `;

        // Animate gauge
        setTimeout(() => {
            const fill = header.querySelector('.conviction-fill');
            if(fill) fill.style.width = conviction + '%';
        }, 500);

        this.generateTags();
    },

    generateTags: function() {
        const text = document.body.innerText.toLowerCase();
        const container = document.getElementById('header-tags');
        if (!container) return;

        container.innerHTML = ''; // Clear previous

        const TAG_RULES = {
            'AI': /ai|artificial intelligence|neural|chatgpt|llm|compute|gpu|nvidia/i,
            'CRYPTO': /crypto|bitcoin|ethereum|btc|eth|blockchain|defi/i,
            'MACRO': /macro|inflation|fed|rates|yield|gdp|cpi|employment|recession/i,
            'ENERGY': /energy|oil|crude|gas|nuclear|power/i,
            'POLICY': /policy|regulation|sec|congress|law|shutdown|geopolitics/i,
            'TECH': /tech|software|saas|cloud|cyber/i,
            'VOLATILITY': /volatility|vix|fear|panic|crash|correction/i
        };

        // 1. Rule-based Tags
        for (const [tag, regex] of Object.entries(TAG_RULES)) {
            if (regex.test(text)) {
                const pill = document.createElement('span');
                pill.className = 'tag-pill';
                pill.textContent = tag;
                pill.onclick = () => this.highlightText(tag);
                container.appendChild(pill);
            }
        }

        // 2. Data-driven Entity Tags (from MOCK_DATA)
        if (this.mockData && this.mockData.credit_memos) {
            const tickers = Object.values(this.mockData.credit_memos).map(m => m.ticker).filter(Boolean);
            const uniqueTickers = [...new Set(tickers)];

            uniqueTickers.forEach(ticker => {
                // Check if ticker exists in text (whole word)
                const tickerRegex = new RegExp(`\\b${ticker}\\b`, 'i');
                if (tickerRegex.test(text)) {
                    const pill = document.createElement('a');
                    pill.className = 'tag-pill concept-link';
                    pill.textContent = ticker;
                    pill.style.borderColor = 'var(--neon-green)';
                    pill.style.color = 'var(--neon-green)';
                    pill.href = `credit_memo_v2.html?ticker=${ticker}`;
                    pill.title = `View Credit Memo for ${ticker}`;
                    container.appendChild(pill);
                }
            });
        }
    },

    processCitations: function() {
        // Priority 1: Existing data-verify-id
        let elements = document.querySelectorAll('[data-verify-id]');

        if (elements.length === 0) {
            // Priority 2: Fallback Regex scanning
            // We need to walk text nodes to avoid breaking HTML
            const walker = document.createTreeWalker(
                document.getElementById('main-content-area'),
                NodeFilter.SHOW_TEXT,
                null,
                false
            );

            let node;
            const nodesToReplace = [];
            // Regex for: Prices ($100.00), Percentages (50%), Years (2020-2030)
            const regex = /(\$\d{1,3}(,\d{3})*(\.\d{2})?)|(\d+(\.\d+)?%)|(20\d{2})/g;

            while(node = walker.nextNode()) {
                // Optimization: Stop scanning once we have enough nodes
                // The processing loop below limits to index <= 20 (21 items)
                if (nodesToReplace.length > 20) break;

                if (node.parentElement.tagName !== 'SCRIPT' &&
                    node.parentElement.tagName !== 'STYLE' &&
                    node.textContent.match(regex)) {
                    nodesToReplace.push(node);
                }
            }

            nodesToReplace.forEach((textNode, index) => {
                // Limit to first 20 to avoid performance hit on large pages
                if (index > 20) return;

                const fragment = document.createDocumentFragment();
                let lastIndex = 0;
                let match;

                // Reset regex
                regex.lastIndex = 0;
                const text = textNode.textContent;

                while ((match = regex.exec(text)) !== null) {
                    // Text before match
                    fragment.appendChild(document.createTextNode(text.substring(lastIndex, match.index)));

                    // The Match
                    const chip = document.createElement('span');
                    chip.className = 'source-chip';
                    chip.textContent = match[0];

                    // Tooltip
                    const tooltip = document.createElement('div');
                    tooltip.className = 'source-tooltip';
                    tooltip.innerHTML = `
                        <div class="tooltip-header">DATA POINT DETECTED</div>
                        <div>VALUE: ${match[0]}</div>
                        <div style="font-size:0.7rem; color:#aaa; margin-top:5px;">Source: Unstructured Text Scan</div>
                        <div style="font-size:0.7rem; color:#00f3ff;">Confidence: ${(Math.random() * 0.2 + 0.8).toFixed(2)}</div>
                    `;
                    chip.appendChild(tooltip);

                    fragment.appendChild(chip);
                    lastIndex = regex.lastIndex;
                }

                // Remaining text
                fragment.appendChild(document.createTextNode(text.substring(lastIndex)));
                textNode.parentNode.replaceChild(fragment, textNode);
            });
            return;
        }

        // If data-verify-id exists, use them
        elements.forEach((el, index) => {
            const verifyId = el.getAttribute('data-verify-id');
            const chip = document.createElement('span');
            chip.className = 'source-chip';
            chip.innerText = `[${index + 1}]`;

            // Check for POS:TICKER pattern for interactive linking
            const tickerMatch = verifyId.match(/POS:([A-Z]+)/);
            if (tickerMatch) {
                const ticker = tickerMatch[1];
                chip.onclick = (e) => {
                    e.stopPropagation();
                    this.logInteraction(`Navigating to intelligence node: ${ticker}`);
                    // Optional: Open in new tab or navigate
                    // window.location.href = `credit_memo_v2.html?ticker=${ticker}`;
                    console.log(`Navigate to ${ticker}`);
                };
                chip.style.cursor = "pointer";
                chip.title = `Click to access ${ticker} Credit Memo`;
            }

            const tooltip = document.createElement('div');
            tooltip.className = 'source-tooltip';
            tooltip.innerHTML = `
                <div class="tooltip-header">VERIFICATION NODE ${index + 1}</div>
                <div>ID: ${verifyId}</div>
                <div style="font-size:0.7rem; color:#aaa; margin-top:5px;">Source: Internal Knowledge Graph</div>
                <div style="font-size:0.7rem; color:#00f3ff;">Hash: ${Math.random().toString(36).substring(7)}</div>
            `;
            chip.appendChild(tooltip);
            el.appendChild(chip);
        });
    },

    injectSystem2: function() {
        const terminal = document.getElementById('sys2-terminal');
        if (!terminal) return;

        terminal.innerHTML = `
            <div class="terminal-header">
                <span>// SYSTEM 2 DIAGNOSTICS</span>
                <span style="cursor:pointer;" onclick="this.parentElement.parentElement.classList.toggle('collapsed')">_</span>
            </div>
            <div id="sys2-logs"></div>
            <div class="scanline"></div>
        `;

        const logs = [
            "Initializing semantic analysis...",
            "Scanning for logical fallacies...",
            "Cross-referencing macro indicators...",
            "Detecting sentiment drift...",
            "CRITIQUE: Thesis relies on forward guidance...",
            "VERDICT: Narrative coherence is HIGH."
        ];

        const logContainer = document.getElementById('sys2-logs');

        logs.forEach((log, i) => {
            setTimeout(() => {
                const entry = document.createElement('div');
                entry.className = 'log-entry';
                const time = new Date().toLocaleTimeString();
                let content = log;
                if (log.startsWith("CRITIQUE:")) content = `<span class="log-critique">${log}</span>`;
                entry.innerHTML = `<span class="log-timestamp">[${time}]</span> ${content}`;
                logContainer.appendChild(entry);
                terminal.scrollTop = terminal.scrollHeight;
            }, i * 800);
        });
    },

    highlightText: function(term) {
        // Clear previous
        this.clearHighlights();
        if (!term) return;

        // Simple find logic (visual only)
        // Using window.find is easy but scroll-jacky.
        // Using TreeWalker to wrap text is better.

        const walker = document.createTreeWalker(
            document.getElementById('main-content-area'),
            NodeFilter.SHOW_TEXT,
            null,
            false
        );

        const regex = new RegExp(term, 'gi');
        const nodesToReplace = [];

        let node;
        while(node = walker.nextNode()) {
            if (node.parentElement.tagName !== 'SCRIPT' &&
                node.parentElement.className !== 'highlight-match' &&
                node.textContent.match(regex)) {
                nodesToReplace.push(node);
            }
        }

        nodesToReplace.forEach(textNode => {
            const fragment = document.createDocumentFragment();
            let lastIndex = 0;
            let match;
            regex.lastIndex = 0;
            const text = textNode.textContent;

            while ((match = regex.exec(text)) !== null) {
                fragment.appendChild(document.createTextNode(text.substring(lastIndex, match.index)));
                const span = document.createElement('span');
                span.className = 'highlight-match';
                span.textContent = match[0];
                fragment.appendChild(span);
                lastIndex = regex.lastIndex;
            }
            fragment.appendChild(document.createTextNode(text.substring(lastIndex)));
            textNode.parentNode.replaceChild(fragment, textNode);
        });
    },

    clearHighlights: function() {
        document.querySelectorAll('.highlight-match').forEach(span => {
            const parent = span.parentNode;
            parent.replaceChild(document.createTextNode(span.textContent), span);
            parent.normalize(); // Merge text nodes
        });
    },

    setupSearch: function() {
        const input = document.getElementById('cyber-search-input');
        if (!input) return;

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                this.highlightText(input.value);
            }
        });
    }
};
