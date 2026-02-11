(function() {
    console.log("Initializing Report Navigation...");

    const MANIFEST_URL = 'data/report_manifest.json';

    async function initNavigation() {
        try {
            // 1. Fetch Manifest
            const response = await fetch(MANIFEST_URL);
            if (!response.ok) throw new Error("Failed to fetch manifest");
            const manifest = await response.json();

            // 2. Identify Current Page
            const path = window.location.pathname;
            const filename = path.split('/').pop();

            const currentItem = manifest.find(item => item.filename === filename);

            if (!currentItem) {
                console.warn("Current page not found in manifest:", filename);
                return;
            }

            // 3. Filter by Type
            // If type is generic "OTHER" or "REPORT", maybe we don't want strict type navigation?
            // For now, strict type navigation is safer to keep context.
            const siblings = manifest.filter(item => item.type === currentItem.type);

            // 4. Find Index (Manifest is sorted Date Descending: Newest [0] -> Oldest [N])
            const currentIndex = siblings.findIndex(item => item.filename === filename);

            if (currentIndex === -1) return;

            // 5. Determine Neighbors
            // "Next" (Newer) is lower index
            const nextItem = currentIndex > 0 ? siblings[currentIndex - 1] : null;

            // "Previous" (Older) is higher index
            const prevItem = currentIndex < siblings.length - 1 ? siblings[currentIndex + 1] : null;

            // 6. Inject UI
            injectNavUI(prevItem, nextItem, currentItem.type);

        } catch (e) {
            console.error("Report Navigation Error:", e);
        }
    }

    function injectNavUI(prev, next, type) {
        // Create Container
        const container = document.createElement('div');
        container.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 20px;
            z-index: 9999;
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            pointer-events: none; /* Let clicks pass through empty space */
        `;

        const btnStyle = `
            background: rgba(0, 0, 0, 0.8);
            color: #00f3ff;
            border: 1px solid #00f3ff;
            padding: 10px 20px;
            text-decoration: none;
            font-size: 12px;
            border-radius: 4px;
            pointer-events: auto;
            transition: all 0.2s;
            backdrop-filter: blur(4px);
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 0 10px rgba(0, 243, 255, 0.1);
        `;

        if (prev) {
            const btn = document.createElement('a');
            btn.href = prev.path;
            btn.innerHTML = `&larr; PREV ${type.replace('_', ' ')}`;
            btn.style.cssText = btnStyle;
            btn.title = `${prev.title} (${prev.date})`;

            btn.onmouseover = () => {
                btn.style.background = 'rgba(0, 243, 255, 0.2)';
                btn.style.boxShadow = '0 0 20px rgba(0, 243, 255, 0.3)';
            };
            btn.onmouseout = () => {
                btn.style.background = 'rgba(0, 0, 0, 0.8)';
                btn.style.boxShadow = '0 0 10px rgba(0, 243, 255, 0.1)';
            };

            container.appendChild(btn);
        }

        // Central Hub Link (Library)
        let hubLink = 'market_mayhem_archive.html';
        if (type === 'DAILY_BRIEFING') hubLink = 'daily_briefings_library.html';
        if (type === 'MARKET_PULSE') hubLink = 'market_pulse_library.html';
        if (type === 'HOUSE_VIEW') hubLink = 'house_view_library.html';

        const hubBtn = document.createElement('a');
        hubBtn.href = hubLink;
        hubBtn.innerHTML = `<i class="fas fa-th"></i>`;
        hubBtn.style.cssText = btnStyle + "padding: 10px 15px;";
        hubBtn.title = "Back to Library";
        hubBtn.onmouseover = () => hubBtn.style.background = 'rgba(255, 255, 255, 0.2)';
        hubBtn.onmouseout = () => hubBtn.style.background = 'rgba(0, 0, 0, 0.8)';
        container.appendChild(hubBtn);

        if (next) {
            const btn = document.createElement('a');
            btn.href = next.path;
            btn.innerHTML = `NEXT ${type.replace('_', ' ')} &rarr;`;
            btn.style.cssText = btnStyle;
            btn.title = `${next.title} (${next.date})`;

            btn.onmouseover = () => {
                btn.style.background = 'rgba(0, 243, 255, 0.2)';
                btn.style.boxShadow = '0 0 20px rgba(0, 243, 255, 0.3)';
            };
            btn.onmouseout = () => {
                btn.style.background = 'rgba(0, 0, 0, 0.8)';
                btn.style.boxShadow = '0 0 10px rgba(0, 243, 255, 0.1)';
            };

            container.appendChild(btn);
        }

        document.body.appendChild(container);
    }

    // Ensure FontAwesome for Hub Icon
    if (!document.querySelector('link[href*="font-awesome"]')) {
        const link = document.createElement('link');
        link.rel = "stylesheet";
        link.href = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css";
        document.head.appendChild(link);
    }

    // Wait for DOM
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initNavigation);
    } else {
        initNavigation();
    }

})();
