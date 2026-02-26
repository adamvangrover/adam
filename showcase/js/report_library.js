/**
 * report_library.js
 *
 * robust, shared logic for rendering report libraries (Market Pulse, Daily Briefing, etc.)
 * supports filtering, sorting, search, pagination, and graceful error handling.
 */

class ReportLibrary {
    constructor(config) {
        this.containerId = config.containerId || 'grid-container';
        this.manifestUrl = config.manifestUrl || 'data/report_manifest.json';
        this.filterType = config.filterType || null; // e.g., 'MARKET_PULSE', or array ['MARKET_PULSE', 'MARKET_MAYHEM']
        this.searchId = config.searchId || 'search-input';
        this.sortId = config.sortId || 'sort-select';

        // Pagination Config
        this.itemsPerPage = 20;
        this.currentPage = 1;

        this.data = [];
        this.filteredData = [];

        this.init();
    }

    async init() {
        this.injectStyles();
        await this.loadData();
        this.setupEventListeners();
        this.render();
    }

    injectStyles() {
        if (document.getElementById('report-library-styles')) return;
        const style = document.createElement('style');
        style.id = 'report-library-styles';
        style.innerHTML = `
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .fade-in {
                animation: fadeIn 0.5s ease-out forwards;
            }
        `;
        document.head.appendChild(style);
    }

    async loadData() {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        try {
            // Redundant Fallback: Try multiple paths if first fails
            // (Assuming script might be run from root or showcase/)
            let response;
            try {
                response = await fetch(this.manifestUrl);
                if (!response.ok) throw new Error('404');
            } catch (e) {
                console.warn(`Primary manifest fetch failed (${this.manifestUrl}), trying fallback...`);
                response = await fetch(`showcase/${this.manifestUrl}`);
            }

            if (!response || !response.ok) throw new Error('Manifest load failed');

            this.data = await response.json();

            // Initial filter by type if configured
            if (this.filterType) {
                if (Array.isArray(this.filterType)) {
                    this.data = this.data.filter(item => this.filterType.includes(item.type));
                } else {
                    this.data = this.data.filter(item => item.type === this.filterType);
                }
            }

            this.filteredData = [...this.data];

        } catch (error) {
            console.error('Library Error:', error);
            this.renderError();
        }
    }

    setupEventListeners() {
        const searchInput = document.getElementById(this.searchId);
        const sortSelect = document.getElementById(this.sortId);

        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.filter(e.target.value);
            });
        }

        if (sortSelect) {
            sortSelect.addEventListener('change', (e) => {
                this.sort(e.target.value);
            });
        }
    }

    filter(query) {
        const lowerQuery = query.toLowerCase();
        this.filteredData = this.data.filter(item => {
            const matchTitle = item.title.toLowerCase().includes(lowerQuery);
            const matchDate = item.date.includes(lowerQuery);
            return matchTitle || matchDate;
        });

        // Reset pagination on filter
        this.currentPage = 1;
        this.render();
    }

    sort(order) {
        // order: 'newest' or 'oldest'
        this.filteredData.sort((a, b) => {
            const dateA = new Date(a.date === 'Unknown' ? '1900-01-01' : a.date);
            const dateB = new Date(b.date === 'Unknown' ? '1900-01-01' : b.date);

            return order === 'oldest' ? dateA - dateB : dateB - dateA;
        });

        // Reset pagination on sort
        this.currentPage = 1;
        this.render();
    }

    loadMore() {
        this.currentPage++;
        this.render(true); // true = append mode
    }

    render(append = false) {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        if (!append) {
            container.innerHTML = '';
        }

        // Remove existing "Load More" button if it exists (so we can move it to bottom)
        const existingBtn = document.getElementById('load-more-btn');
        if (existingBtn) existingBtn.remove();

        if (this.filteredData.length === 0) {
            container.innerHTML = `
                <div class="col-span-full text-center py-20">
                    <div class="text-gray-400 text-lg mb-2">No reports found matching criteria.</div>
                    <div class="text-gray-500 text-sm">Try adjusting your filters or search query.</div>
                </div>
            `;
            return;
        }

        const start = (this.currentPage - 1) * this.itemsPerPage;
        const end = start + this.itemsPerPage;
        const pageItems = this.filteredData.slice(start, end);

        pageItems.forEach((item, index) => {
            const card = document.createElement('a');
            card.href = item.path;
            card.className = 'paper-card block p-6 rounded-lg group text-left relative overflow-hidden bg-white hover:bg-gray-50 transition-all duration-300 border border-gray-200 fade-in';
            card.style.animationDelay = `${index * 50}ms`; // Staggered animation

            // Badge color logic
            let badgeColor = 'bg-gray-100 text-gray-800 border-gray-200';
            if (item.type === 'MARKET_MAYHEM') badgeColor = 'bg-red-100 text-red-800 border-red-200';
            if (item.type === 'MARKET_PULSE') badgeColor = 'bg-blue-100 text-blue-800 border-blue-200';
            if (item.type === 'DAILY_BRIEFING') badgeColor = 'bg-green-100 text-green-800 border-green-200';
            if (item.type === 'HOUSE_VIEW') badgeColor = 'bg-purple-100 text-purple-800 border-purple-200';
            if (item.type === 'DEEP_DIVE') badgeColor = 'bg-indigo-100 text-indigo-800 border-indigo-200';

            const displayDate = item.date === 'Unknown' ? 'ARCHIVED' : item.date;

            // Check for description or summary
            const description = item.description || item.summary || 'CONFIDENTIAL // ADAM SYSTEM GENERATED REPORT';

            card.innerHTML = `
                <div class="flex justify-between items-start mb-4">
                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${badgeColor}">
                        ${item.type.replace('_', ' ')}
                    </span>
                    <span class="text-xs text-gray-400 font-mono flex items-center gap-1">
                        <i class="far fa-clock"></i> ${displayDate}
                    </span>
                </div>

                <h3 class="text-xl font-bold serif text-gray-900 mb-3 group-hover:text-blue-600 transition-colors leading-tight">
                    ${item.title}
                </h3>

                <div class="text-sm text-gray-500 line-clamp-2 mb-4">
                    ${description}
                </div>

                <div class="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-gray-200 to-transparent group-hover:via-blue-400 transition-all duration-500"></div>
            `;
            container.appendChild(card);
        });

        // Add "Load More" button if there are more items
        if (this.filteredData.length > end) {
            const btnContainer = document.createElement('div');
            btnContainer.id = 'load-more-btn';
            btnContainer.className = 'col-span-full flex justify-center py-8';
            btnContainer.innerHTML = `
                <button class="px-6 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-md font-medium text-sm transition-colors">
                    Load More Reports (${this.filteredData.length - end} remaining)
                </button>
            `;
            btnContainer.querySelector('button').addEventListener('click', () => {
                this.loadMore();
            });
            container.appendChild(btnContainer);
        }

        // Update stats if element exists
        const countEl = document.getElementById('report-count');
        if (countEl) countEl.innerText = `${this.filteredData.length} Reports`;
    }

    renderError() {
        const container = document.getElementById(this.containerId);
        if (container) {
            container.innerHTML = `
                <div class="col-span-full text-center py-20">
                    <div class="text-red-500 text-lg font-bold mb-2">System Error</div>
                    <div class="text-gray-600">Unable to access Intelligence Library Manifest.</div>
                    <div class="text-xs text-gray-400 mt-4">Error Code: MANIFEST_FETCH_FAILURE</div>
                </div>
            `;
        }
    }
}

// Global exposure for inline init
window.ReportLibrary = ReportLibrary;
