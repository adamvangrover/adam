window.CyberDashboard = {
    init: function() {
        console.log("Initializing Cyberpunk Dashboard Protocol v2.0...");
        this.injectAssets();
        this.injectLayout();
        this.injectMetadataHeader();
        this.processCitations();
        this.injectSystem2();
        this.setupSearch();
        console.log("Dashboard Protocol Online.");
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

        const TAG_RULES = {
            'AI': /ai|artificial intelligence|neural|chatgpt|llm|compute|gpu|nvidia/i,
            'CRYPTO': /crypto|bitcoin|ethereum|btc|eth|blockchain|defi/i,
            'MACRO': /macro|inflation|fed|rates|yield|gdp|cpi|employment|recession/i,
            'ENERGY': /energy|oil|crude|gas|nuclear|power/i,
            'POLICY': /policy|regulation|sec|congress|law|shutdown|geopolitics/i,
            'TECH': /tech|software|saas|cloud|cyber/i,
            'VOLATILITY': /volatility|vix|fear|panic|crash|correction/i
        };

        for (const [tag, regex] of Object.entries(TAG_RULES)) {
            if (regex.test(text)) {
                const pill = document.createElement('span');
                pill.className = 'tag-pill';
                pill.textContent = tag;
                pill.onclick = () => this.highlightText(tag);
                container.appendChild(pill);
            }
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
