document.addEventListener('DOMContentLoaded', () => {
    // 1. Inject Toggle Button
    const toggleBtn = document.createElement('button');
    toggleBtn.textContent = 'VERIFY DATA';
    toggleBtn.className = 'cyber-btn';
    toggleBtn.style.position = 'fixed';
    toggleBtn.style.bottom = '20px';
    toggleBtn.style.right = '20px';
    toggleBtn.style.zIndex = '9999';
    toggleBtn.style.backgroundColor = '#000';
    toggleBtn.style.border = '1px solid #00f3ff';
    toggleBtn.style.color = '#00f3ff';
    toggleBtn.style.fontFamily = 'JetBrains Mono';
    toggleBtn.style.cursor = 'pointer';
    toggleBtn.style.padding = '10px 20px';
    toggleBtn.style.boxShadow = '0 0 10px rgba(0, 243, 255, 0.3)';

    document.body.appendChild(toggleBtn);

    let isVerified = false;
    let contextData = null;

    // 2. Fetch Data
    async function loadData() {
        try {
            // In a real app this would be an API call, here we load the static JSON
            // We assume the file is served correctly relative to the HTML
            // Note: Since this is often file:// or local, we might need to mock if fetch fails
            // For this implementation, we will try to fetch, else fallback (or assume verification script mocks it)
            // But since we are "building additive", let's assume standard fetch works for the repo structure
            // We need to point to the file we just created.
            // Relative path from showcase/newsletter...html to core/data/snapshots...
            // ../core/data/snapshots/jan30_2026_context.json
            const response = await fetch('../core/data/snapshots/jan30_2026_context.json');
            contextData = await response.json();
        } catch (e) {
            console.error("Failed to load context data:", e);
            // Fallback for simple local viewing if fetch is blocked by CORS/Protocol
            return;
        }
    }

    loadData();

    // 3. Toggle Logic
    toggleBtn.addEventListener('click', () => {
        isVerified = !isVerified;
        toggleBtn.textContent = isVerified ? 'VERIFIED MODE: ON' : 'VERIFY DATA';
        toggleBtn.style.backgroundColor = isVerified ? '#00f3ff' : '#000';
        toggleBtn.style.color = isVerified ? '#000' : '#00f3ff';

        if (isVerified && contextData) {
            enableOverlay();
        } else {
            disableOverlay();
        }
    });

    function enableOverlay() {
        const elements = document.querySelectorAll('[data-verify-id]');
        elements.forEach(el => {
            const id = el.getAttribute('data-verify-id');
            const data = resolveData(id);

            if (data) {
                el.style.position = 'relative';
                el.style.borderBottom = '2px dashed #00f3ff';
                el.style.cursor = 'help';

                // Create Tooltip Logic
                el.onmouseenter = (e) => showTooltip(e, data);
                el.onmouseleave = () => hideTooltip();
            }
        });
    }

    function disableOverlay() {
        const elements = document.querySelectorAll('[data-verify-id]');
        elements.forEach(el => {
            el.style.borderBottom = 'none';
            el.style.cursor = 'default';
            el.onmouseenter = null;
            el.onmouseleave = null;
        });
    }

    function resolveData(id) {
        if (!contextData) return null;
        if (id.startsWith('POS:')) {
            const ticker = id.split(':')[1];
            return contextData.positions[ticker];
        }
        if (id.startsWith('THEME:')) {
            const theme = id.split(':')[1];
            return contextData.themes[theme];
        }
        return null;
    }

    // Tooltip Implementation
    let tooltip = null;

    function showTooltip(e, data) {
        if (tooltip) tooltip.remove();

        tooltip = document.createElement('div');
        tooltip.style.position = 'absolute';
        tooltip.style.left = `${e.pageX + 15}px`;
        tooltip.style.top = `${e.pageY + 15}px`;
        tooltip.style.background = 'rgba(0, 10, 20, 0.95)';
        tooltip.style.border = '1px solid #00f3ff';
        tooltip.style.padding = '15px';
        tooltip.style.color = '#fff';
        tooltip.style.fontFamily = 'JetBrains Mono';
        tooltip.style.fontSize = '0.8rem';
        tooltip.style.maxWidth = '300px';
        tooltip.style.zIndex = '10000';
        tooltip.style.boxShadow = '0 0 20px rgba(0, 243, 255, 0.2)';
        tooltip.style.pointerEvents = 'none';

        let content = `<strong>// SYSTEM VERIFIED</strong><br>`;

        if (data.ticker) {
            content += `<span style="color:#00f3ff">TICKER: ${data.ticker}</span><br>`;
            content += `CONVICTION: ${data.conviction}<br>`;
            content += `TARGET: ${data.target}<br>`;
            content += `<hr style="border:0; border-top:1px solid #333; margin:8px 0;">`;
            content += `<div style="opacity:0.8; font-size:0.75rem;">LOGS:<br>`;
            data.consensus_log.forEach(log => {
                content += `> ${log}<br>`;
            });
            content += `</div>`;
        } else if (data.name) {
             content += `<span style="color:#f0f">${data.name}</span><br>`;
             content += `IMPACT: ${data.impact}<br>`;
             content += `<div style="margin-top:5px; opacity:0.9;">"${data.logic}"</div>`;
        }

        tooltip.innerHTML = content;
        document.body.appendChild(tooltip);
    }

    function hideTooltip() {
        if (tooltip) tooltip.remove();
        tooltip = null;
    }
});
