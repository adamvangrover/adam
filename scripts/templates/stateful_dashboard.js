/**
 * Stateful Dashboard Template
 * Uses StateManager for persistence and ModularLoader for data.
 */

// Ensure dependencies
if (!window.stateManager || !window.ModularLoader) {
    console.error("Dependencies missing: StateManager or ModularLoader");
}

const DASHBOARD_ID = "{{DASHBOARD_ID}}";

class StatefulDashboard {
    constructor() {
        this.loader = new window.ModularLoader('data/modular/');
        this.sm = window.stateManager;

        // Initialize State
        this.sm.init(`${DASHBOARD_ID}_filter`, 'all', true);
        this.sm.init(`${DASHBOARD_ID}_view`, 'grid', true);

        this.bindEvents();
        this.render();
    }

    bindEvents() {
        // Subscribe to state changes
        this.sm.subscribe(`${DASHBOARD_ID}_filter`, (val) => this.onFilterChange(val));
        this.sm.subscribe(`${DASHBOARD_ID}_view`, (val) => this.onViewChange(val));
    }

    async loadData() {
        // Prefetch common modules
        this.loader.prefetch(['reports', 'market_data']);

        // Load primary data
        const data = await this.loader.load('reports');
        return data;
    }

    onFilterChange(filter) {
        console.log(`Filter changed to: ${filter}`);
        this.render();
    }

    onViewChange(view) {
        console.log(`View changed to: ${view}`);
        this.render();
    }

    render() {
        const filter = this.sm.getState(`${DASHBOARD_ID}_filter`);
        const view = this.sm.getState(`${DASHBOARD_ID}_view`);

        // Logic to update DOM based on state
        document.body.setAttribute('data-view', view);
        // ...
    }
}

// Initialize
// window.dashboard = new StatefulDashboard();
