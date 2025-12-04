// Navigation Manager
class NavManager {
    constructor() {
        this.currentPath = window.location.pathname.split('/').pop() || 'index.html';
        this.navToggle = document.getElementById('nav-toggle');
        this.mobileMenu = null;

        this.init();
    }

    init() {
        // Highlight active link if we had a sidebar
        // Since most pages are standalone, we just handle the mobile toggle
        if (this.navToggle) {
            this.navToggle.addEventListener('click', () => this.toggleMobileMenu());
        }
    }

    toggleMobileMenu() {
        if (!this.mobileMenu) {
            this.createMobileMenu();
        }
        this.mobileMenu.classList.toggle('hidden');
    }

    createMobileMenu() {
        this.mobileMenu = document.createElement('div');
        this.mobileMenu.className = 'fixed inset-0 bg-slate-900/95 z-40 flex flex-col items-center justify-center space-y-6 hidden transition-all duration-300 backdrop-blur-md';

        const links = [
            { name: 'Mission Control', href: 'index.html' },
            { name: 'Neural Dashboard', href: 'neural_dashboard.html' },
            { name: 'Deep Dive', href: 'deep_dive.html' },
            { name: 'Financial Twin', href: 'financial_twin.html' },
            { name: 'Agent Registry', href: 'agents.html' },
            { name: 'The Vault', href: 'data.html' }
        ];

        links.forEach(link => {
            const a = document.createElement('a');
            a.href = link.href;
            a.className = `text-2xl font-mono font-bold hover:text-cyan-400 transition ${this.currentPath === link.href ? 'text-cyan-400' : 'text-slate-400'}`;
            a.textContent = link.name;
            this.mobileMenu.appendChild(a);
        });

        // Close button
        const closeBtn = document.createElement('button');
        closeBtn.className = 'absolute top-6 right-6 text-slate-400 hover:text-white';
        closeBtn.innerHTML = '<i class="fas fa-times text-2xl"></i>';
        closeBtn.onclick = () => this.toggleMobileMenu();
        this.mobileMenu.appendChild(closeBtn);

        document.body.appendChild(this.mobileMenu);
    }
}

// Init on load
document.addEventListener('DOMContentLoaded', () => {
    window.navManager = new NavManager();
});
