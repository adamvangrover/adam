import axios from 'axios';

const API_BASE_URL = 'http://localhost:5001/api';

const ENDPOINT_MAPPING: Record<string, string> = {
  '/market/data': '/data/adam_market_baseline.json',
  '/graph/data': '/data/knowledge_graph.json',
  '/agents/status': '/data/agents_status.json',
  '/system/health': '/data/system_health.json',
  '/news/feed': '/data/news_feed.json',
  '/manifest': '/data/manifest.json'
};

class DataManager {
  private useFallback: boolean = false;
  private connectionChecked: boolean = false;
  private manifest: any = null;

  async checkConnection() {
    if (this.connectionChecked) return;

    try {
      await axios.get(`${API_BASE_URL}/health`, { timeout: 2000 });
      this.useFallback = false;
      console.log('ADAM CORE :: ONLINE');
    } catch (error) {
      this.useFallback = true;
      console.log('ADAM CORE :: OFFLINE (SIMULATION MODE ACTIVE)');
    } finally {
      this.connectionChecked = true;
    }
  }

  isOfflineMode() {
    return this.useFallback;
  }

  async getManifest() {
    if (this.manifest) return this.manifest;
    try {
        const response = await fetch(ENDPOINT_MAPPING['/manifest']);
        if (response.ok) {
            this.manifest = await response.json();
            return this.manifest;
        }
    } catch (e) {
        console.error("Failed to load manifest", e);
    }
    return { agents: [], reports: [], strategies: [] };
  }

  async getData(endpoint: string) {
    await this.checkConnection();

    if (this.useFallback) {
      return this.simulateResponse(endpoint);
    }

    try {
      const response = await axios.get(`${API_BASE_URL}${endpoint}`);
      return response.data;
    } catch (error) {
      console.warn(`API Failed: ${endpoint}. Switching to Simulation.`);
      this.useFallback = true;
      return this.simulateResponse(endpoint);
    }
  }

  private async simulateResponse(endpoint: string) {
    // 1. Try to load static file if exists
    if (ENDPOINT_MAPPING[endpoint]) {
        try {
            const response = await fetch(ENDPOINT_MAPPING[endpoint]);
            if (response.ok) return await response.json();
        } catch (e) { /* ignore */ }
    }

    // 2. Dynamic Simulation (The "Simulation Engine")
    console.log(`Simulating data for ${endpoint}`);

    if (endpoint.includes('market')) {
        return this.generateMockMarketData();
    }
    if (endpoint.includes('agents')) {
        const manifest = await this.getManifest();
        return manifest.agents || [];
    }
    if (endpoint.includes('reports')) {
        const manifest = await this.getManifest();
        return manifest.reports || [];
    }

    // Default mock response
    return { status: "simulated", timestamp: Date.now() };
  }

  private generateMockMarketData() {
    const tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'GS'];
    return tickers.map(t => ({
        ticker: t,
        price: (100 + Math.random() * 900).toFixed(2),
        change: (Math.random() * 10 - 5).toFixed(2),
        volume: Math.floor(Math.random() * 10000000)
    }));
  }
}

export const dataManager = new DataManager();
