// Command Palette Logic
(function() {
    // Styles
    const style = document.createElement('style');
    style.innerHTML = `
        #cmd-palette-overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.8); backdrop-filter: blur(5px);
            z-index: 10000; display: none; align-items: flex-start; justify-content: center;
            padding-top: 100px;
        }
        #cmd-palette {
            width: 600px; max-width: 90%; background: #0a0f16; border: 1px solid #333;
            box-shadow: 0 0 50px rgba(0, 243, 255, 0.1); border-radius: 8px; overflow: hidden;
            font-family: 'Inter', sans-serif;
        }
        #cmd-input {
            width: 100%; padding: 20px; background: transparent; border: none; border-bottom: 1px solid #333;
            color: #fff; font-size: 1.2rem; outline: none; font-family: 'JetBrains Mono', monospace;
        }
        #cmd-results {
            max-height: 400px; overflow-y: auto;
        }
        .cmd-item {
            padding: 15px 20px; border-bottom: 1px solid #222; cursor: pointer; display: flex; align-items: center; justify-content: space-between;
        }
        .cmd-item:hover, .cmd-item.active {
            background: rgba(0, 243, 255, 0.1); border-left: 3px solid #00f3ff;
        }
        .cmd-icon { margin-right: 15px; font-size: 1.2rem; width: 20px; text-align: center; }
        .cmd-text { flex-grow: 1; }
        .cmd-tag { font-size: 0.7rem; background: #333; padding: 2px 6px; border-radius: 4px; color: #aaa; font-family: 'JetBrains Mono'; }
        .cmd-shortcut { font-size: 0.8rem; color: #666; margin-left: 10px; font-family: 'JetBrains Mono'; }
    `;
    document.head.appendChild(style);

    // HTML Structure
    const overlay = document.createElement('div');
    overlay.id = 'cmd-palette-overlay';
    overlay.innerHTML = `
        <div id="cmd-palette">
            <input type="text" id="cmd-input" placeholder="> Type a command or search..." autocomplete="off">
            <div id="cmd-results"></div>
        </div>
    `;
    document.body.appendChild(overlay);

    const input = document.getElementById('cmd-input');
    const results = document.getElementById('cmd-results');

    // Data Sources
    const pages = [
        { title: "Mission Control", url: "index.html", icon: "🏠", tag: "PAGE" },
        { title: "Agent Registry", url: "agents.html", icon: "👥", tag: "PAGE" },
        { title: "Market Mayhem Archive", url: "market_mayhem_archive.html", icon: "📉", tag: "PAGE" },
        { title: "System Evolution", url: "evolution.html", icon: "🧬", tag: "PAGE" },
        { title: "Deep Dive Analyst", url: "deep_dive.html", icon: "🧠", tag: "TOOL" },
        { title: "Deployment Console", url: "deployment.html", icon: "⚙️", tag: "TOOL" },
        { title: "Neural Dashboard", url: "neural_dashboard.html", icon: "📊", tag: "PAGE" }
    ];

    let searchData = [...pages];

    // Fetch Archive Data if available
    if (window.ADAM_ARCHIVE_DATA) {
        // Pre-loaded from another script
    } else {
        // Attempt to load archive.json logic or scrape existing links if on archive page
        // For now, we will add a few hardcoded shortcuts for demo
        searchData.push({ title: "Report: Reflationary Boom", url: "report_reflationary_boom.html", icon: "📄", tag: "REPORT" });
        searchData.push({ title: "Report: MSFT Deep Dive", url: "report_msft_company_report.html", icon: "📄", tag: "REPORT" });
    }

    // Event Listeners
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            togglePalette();
        }
        if (e.key === 'Escape' && overlay.style.display === 'flex') {
            togglePalette();
        }
    });

    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) togglePalette();
    });

    input.addEventListener('input', (e) => {
        renderResults(e.target.value);
    });

    function togglePalette() {
        if (overlay.style.display === 'flex') {
            overlay.style.display = 'none';
        } else {
            overlay.style.display = 'flex';
            input.value = '';
            input.focus();
            renderResults('');
        }
    }

    function renderResults(query) {
        results.innerHTML = '';
        const q = query.toLowerCase();

        const filtered = searchData.filter(item =>
            item.title.toLowerCase().includes(q) ||
            item.tag.toLowerCase().includes(q)
        );

        filtered.forEach(item => {
            const div = document.createElement('div');
            div.className = 'cmd-item';
            div.innerHTML = `
                <span class="cmd-icon">${item.icon}</span>
                <span class="cmd-text">${item.title}</span>
                <span class="cmd-tag">${item.tag}</span>
            `;
            div.onclick = () => window.location.href = item.url;
            results.appendChild(div);
        });

        if (filtered.length === 0) {
             results.innerHTML = '<div style="padding:20px; color:#666; text-align:center;">No results found.</div>';
        }
    }

    // Hint visual
    const hint = document.createElement('div');
    hint.innerHTML = 'Press <span style="background:#333; padding:2px 4px; border-radius:4px; font-family:monospace;">Ctrl+K</span> for commands';
    hint.style.cssText = 'position:fixed; bottom:20px; right:20px; color:#444; font-size:0.8rem; z-index:9000; pointer-events:none;';
    document.body.appendChild(hint);

})();
