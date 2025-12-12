import axios from 'axios';

// Configuration
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000/api';
const MOCK_DELAY = 500; // ms

// Endpoints
const ENDPOINTS = {
    STATUS: '/status',
    AGENTS: '/agents',
    KNOWLEDGE_GRAPH: '/knowledge-graph',
    MARKET_DATA: '/market/data',
    LOGS: '/logs',
    SIMULATION: '/simulation/run'
};

interface ConnectionStatus {
    status: string;
    latency: number;
}

class DataManager {
    mode: 'LIVE' | 'ARCHIVE';
    manifest: any;
    mockData: any;

    constructor() {
        this.mode = 'LIVE'; // 'LIVE' or 'ARCHIVE' (Simulated)
        this.manifest = null;
        this.mockData = {};
    }

    setMode(mode: 'LIVE' | 'ARCHIVE') {
        this.mode = mode;
        console.log(`[DataManager] Switched to ${mode} mode.`);
    }

    async checkConnection(): Promise<ConnectionStatus> {
        if (this.mode === 'ARCHIVE') {
            return { status: 'SIMULATED', latency: 0 };
        }
        try {
            const start = Date.now();
            // Assuming hello endpoint is lighter check, or create a simple status check
            await axios.get(`${API_BASE_URL}/hello`, { timeout: 2000 });
            const latency = Date.now() - start;
            return { status: 'ONLINE', latency };
        } catch (error) {
            console.warn('[DataManager] Backend unreachable, falling back to SIMULATION.');
            return { status: 'OFFLINE (Simulated)', latency: -1 };
        }
    }

    async getManifest() {
        if (this.manifest) return this.manifest;
        try {
            const response = await axios.get('/data/manifest.json');
            this.manifest = response.data;
            return this.manifest;
        } catch (error) {
            console.error('[DataManager] Failed to load manifest.json', error);
            return { agents: [], reports: [], knowledge_graph: {} };
        }
    }

    // Generic fetcher with fallback
    async fetchData(endpoint: string, options = {}) {
        if (this.mode === 'LIVE') {
            try {
                const response = await axios.get(`${API_BASE_URL}${endpoint}`, options);
                return response.data;
            } catch (error) {
                console.warn(`[DataManager] API call to ${endpoint} failed. Using simulation.`);
            }
        }
        return this.getSimulatedData(endpoint);
    }

    async getSimulatedData(endpoint: string) {
        await new Promise(resolve => setTimeout(resolve, MOCK_DELAY));

        switch (endpoint) {
            case ENDPOINTS.STATUS:
                return { status: 'ok', version: 'v23.5.0' };
            case ENDPOINTS.AGENTS:
                const manifest = await this.getManifest();
                return manifest.agents || [];
            case ENDPOINTS.MARKET_DATA:
                return this.generateMockMarketData();
            default:
                return { error: 'No simulation data available for this endpoint' };
        }
    }

    generateMockMarketData() {
        const tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'];
        return tickers.map(t => ({
            ticker: t,
            price: (Math.random() * 200 + 100).toFixed(2),
            change: (Math.random() * 10 - 5).toFixed(2)
        }));
    }
}

export const dataManager = new DataManager();
