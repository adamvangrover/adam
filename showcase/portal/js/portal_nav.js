/**
 * ADAM NEXT-GEN PORTAL NAVIGATION
 * Dynamically builds the sidebar from site_map.json and handles interactivity.
 */

class PortalNavigator {
    constructor() {
        this.siteMapUrl = '../site_map.json';
        this.siteMapData = null;
        this.currentPath = window.location.pathname.split('/').pop() || 'index.html';

        // DOM Elements
        this.sidebar = document.getElementById('portal-sidebar');
        this.navContainer = document.getElementById('sidebar-nav');
        this.toggleBtn = document.getElementById('sidebar-toggle');
        this.mobileToggleBtn = document.getElementById('mobile-toggle');

        // State
        this.isCollapsed = localStorage.getItem('portal_sidebar_collapsed') === 'true';
    }

    async init() {
        if (!this.sidebar || !this.navContainer) {
            console.error('PortalNavigator: Required DOM elements not found.');
            return;
        }

        this.bindEvents();
        this.applyInitialState();

        try {
            await this.loadSiteMap();
            this.renderNavigation();
            this.updateActiveState();
        } catch (error) {
            console.error('PortalNavigator: Failed to initialize.', error);
            this.renderErrorState();
        }
    }

    bindEvents() {
        if (this.toggleBtn) {
            this.toggleBtn.addEventListener('click', () => this.toggleSidebar());
        }

        if (this.mobileToggleBtn) {
            this.mobileToggleBtn.addEventListener('click', () => {
                this.sidebar.classList.toggle('mobile-open');
            });
        }

        // Close sidebar on mobile when clicking outside
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768 &&
                this.sidebar.classList.contains('mobile-open') &&
                !this.sidebar.contains(e.target) &&
                e.target !== this.mobileToggleBtn) {
                this.sidebar.classList.remove('mobile-open');
            }
        });
    }

    applyInitialState() {
        if (this.isCollapsed && window.innerWidth > 768) {
            this.sidebar.classList.add('collapsed');
        }
    }

    toggleSidebar() {
        this.isCollapsed = !this.isCollapsed;
        this.sidebar.classList.toggle('collapsed', this.isCollapsed);
        localStorage.setItem('portal_sidebar_collapsed', this.isCollapsed);
    }

    async loadSiteMap() {
        const response = await fetch(this.siteMapUrl);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        this.siteMapData = await response.json();
    }

    renderNavigation() {
        this.navContainer.innerHTML = '';

        // Add 'Home / Portal' link at the top
        const homeGroup = document.createElement('div');
        homeGroup.className = 'nav-group';
        homeGroup.innerHTML = `
            <a href="index.html" class="nav-item ${this.currentPath === 'index.html' ? 'active' : ''}">
                <i class="fas fa-home nav-icon"></i>
                <span class="nav-text">Portal Home</span>
                <div class="nav-tooltip">Portal Home</div>
            </a>
            <a href="analytics.html" class="nav-item ${this.currentPath === 'analytics.html' ? 'active' : ''}">
                <i class="fas fa-chart-network nav-icon"></i>
                <span class="nav-text">Analytics</span>
                <div class="nav-tooltip">Analytics</div>
            </a>
            <a href="intelligence.html" class="nav-item ${this.currentPath === 'intelligence.html' ? 'active' : ''}">
                <i class="fas fa-brain nav-icon"></i>
                <span class="nav-text">Intelligence</span>
                <div class="nav-tooltip">Intelligence</div>
            </a>
             <a href="../index.html" class="nav-item">
                <i class="fas fa-arrow-left nav-icon"></i>
                <span class="nav-text">Back to Legacy</span>
                <div class="nav-tooltip">Back to Legacy</div>
            </a>
        `;
        this.navContainer.appendChild(homeGroup);

        // Render dynamic categories
        if (this.siteMapData && this.siteMapData.categories) {
            this.siteMapData.categories.forEach(category => {
                const group = document.createElement('div');
                group.className = 'nav-group';

                const title = document.createElement('div');
                title.className = 'nav-group-title';
                title.textContent = category.name;
                group.appendChild(title);

                category.items.forEach(item => {
                    const link = document.createElement('a');
                    // Adjust path to point to parent directory since we are in /portal/
                    link.href = `../${item.link}`;
                    link.className = 'nav-item';

                    // Simple path matching for active state
                    if (this.currentPath !== 'index.html' &&
                        this.currentPath !== 'analytics.html' &&
                        this.currentPath !== 'intelligence.html' &&
                        item.link.includes(this.currentPath)) {
                        link.classList.add('active');
                    }

                    link.innerHTML = `
                        <i class="fas ${item.icon || 'fa-link'} nav-icon"></i>
                        <span class="nav-text">${item.name}</span>
                        <div class="nav-tooltip">${item.name}</div>
                    `;

                    group.appendChild(link);
                });

                this.navContainer.appendChild(group);
            });
        }
    }

    updateActiveState() {
        // Active state logic is mostly handled during rendering for simplicity in this static setup
    }

    renderErrorState() {
        this.navContainer.innerHTML = `
            <div style="padding: 20px; color: #ef4444; font-size: 0.8rem; text-align: center;">
                <i class="fas fa-exclamation-triangle" style="font-size: 2rem; margin-bottom: 10px;"></i><br>
                Failed to load navigation map.
            </div>
        `;
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const navigator = new PortalNavigator();
    navigator.init();

    // Mock Dashboard Data Updates (if on dashboard page)
    if (document.getElementById('mock-vix')) {
        setInterval(() => {
            const vixEl = document.getElementById('mock-vix');
            const current = parseFloat(vixEl.innerText);
            const change = (Math.random() - 0.5) * 0.5;
            vixEl.innerText = (current + change).toFixed(2);
        }, 3000);
    }
});
