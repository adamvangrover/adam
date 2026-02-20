/**
 * ADAM v23.5 SWARM EXTENSION NAVIGATOR
 * -----------------------------------------------------------------------------
 * Additive module to inject "Swarm" features into the main navigation.
 * -----------------------------------------------------------------------------
 */

(function() {
    console.log("[NavSwarm] Initializing Swarm Extensions...");

    function injectSwarmNav() {
        const navList = document.querySelector('#side-nav ul');
        if (!navList) {
            // If nav isn't ready, retry in 100ms
            setTimeout(injectSwarmNav, 100);
            return;
        }

        // Check if already injected
        if (document.getElementById('nav-swarm-header')) return;

        // Divider
        const divider = document.createElement('li');
        divider.className = "my-2 border-t border-slate-800";
        navList.appendChild(divider);

        // Header
        // const header = document.createElement('li');
        // header.id = "nav-swarm-header";
        // header.className = "px-3 py-2 text-xs font-bold text-slate-500 uppercase tracking-wider";
        // header.innerText = "Swarm Extensions";
        // navList.appendChild(header);

        // Items
        const items = [
            { name: "Mayhem Builder", link: "market_mayhem_builder.html", icon: "fa-hammer", color: "text-amber-400" },
            { name: "Sovereign Dash (Live)", link: "sovereign_dashboard.html", icon: "fa-globe-americas", color: "text-emerald-400" },
            { name: "Credit Auto (Enhanced)", link: "credit_memo_automation.html", icon: "fa-robot", color: "text-blue-400" },
            { name: "Archive Search", link: "market_mayhem_archive.html", icon: "fa-search", color: "text-purple-400" }
        ];

        items.forEach(item => {
            const li = document.createElement('li');
            const isActive = window.location.pathname.includes(item.link);
            const activeClass = isActive
                ? 'bg-amber-900/20 text-amber-400 border-l-2 border-amber-400'
                : 'text-slate-400 hover:bg-slate-800 hover:text-white border-l-2 border-transparent';

            li.innerHTML = `
                <a href="${item.link}" class="flex items-center gap-3 px-3 py-2 rounded text-sm font-medium transition group ${activeClass}">
                    <i class="fas ${item.icon} w-5 text-center ${item.color} group-hover:text-white transition"></i>
                    ${item.name}
                </a>
            `;
            navList.appendChild(li);
        });

        console.log("[NavSwarm] Extensions Injected.");
    }

    // Start observing or polling
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectSwarmNav);
    } else {
        injectSwarmNav();
    }

})();
