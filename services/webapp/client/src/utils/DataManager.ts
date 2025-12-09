import axios from 'axios';

// Types for better safety
interface SystemStatus {
  status: 'ONLINE' | 'OFFLINE' | 'SIMULATED';
  latency: number;
  version: string;
}

interface DataManifest {
  reports: Array<{ id: string; title: string; date: string; type: string; path: string }>;
  agents: Array<{ id: string; name: string; status: string; specialization: string }>;
}

const API_BASE_URL = 'http://localhost:5001/api';

class DataManager {
  private useFallback: boolean = false;
  private connectionChecked: boolean = false;
  private manifestCache: DataManifest | null = null;

  constructor() {
    // Initialize simulation data if needed
  }

  async checkConnection(): Promise<SystemStatus> {
    try {
      const start = Date.now();
      await axios.get(`${API_BASE_URL}/health`, { timeout: 2000 });
      this.useFallback = false;
      return { status: 'ONLINE', latency: Date.now() - start, version: 'v23.5' };
    } catch (error) {
      this.useFallback = true;
      return { status: 'SIMULATED', latency: 0, version: 'v23.5 (Offline)' };
    } finally {
      this.connectionChecked = true;
    }
  }

  // Phase 2: Manifest Crawler Simulation
  async getManifest(): Promise<DataManifest> {
    if (this.useFallback || !this.manifestCache) {
      // Simulate crawling core/libraries_and_archives
      this.manifestCache = {
        reports: [
          { id: 'NVDA_2025', title: 'NVDA Deep Dive', date: '2025-02-26', type: 'JSON', path: '/data/nvda.json' },
          { id: 'MM_OCT', title: 'Market Mayhem: October', date: '2025-10-31', type: 'Markdown', path: '/newsletters/mm_oct.md' },
          { id: 'CREDIT_AAPL', title: 'Apple Credit Assessment', date: '2025-03-03', type: 'SNC', path: '/reports/aapl_snc.json' }
        ],
        agents: [
          { id: 'risk_arch', name: 'Risk Architect', status: 'Active', specialization: 'Systemic Risk' },
          { id: 'market_sent', name: 'Sentiment Engine', status: 'Idle', specialization: 'NLP Analysis' },
          { id: 'quant_core', name: 'Quantum Forecaster', status: 'Processing', specialization: 'Predictive Modeling' }
        ]
      };
    }
    return this.manifestCache;
  }

  async getData(endpoint: string): Promise<any> {
    await this.checkConnection();

    if (this.useFallback) {
      return this.simulateEndpoint(endpoint);
    }

    try {
      const response = await axios.get(`${API_BASE_URL}${endpoint}`);
      return response.data;
    } catch (error) {
      console.warn(`Live fetch failed for ${endpoint}, switching to simulation.`);
      return this.simulateEndpoint(endpoint);
    }
  }

  // Phase 2: Simulation Engine
  private simulateEndpoint(endpoint: string): any {
    if (endpoint.includes('market')) {
      // Return slightly randomized market data
      return {
        SPY: 512.45 + (Math.random() * 2 - 1),
        BTC: 68500 + (Math.random() * 100 - 50),
        VIX: 14.2 + (Math.random() * 0.5)
      };
    }
    if (endpoint.includes('system/health')) {
      return { cpu: 45, memory: 62, active_threads: 12 };
    }
    return {};
  }

  isOfflineMode() { return this.useFallback; }
}

export const dataManager = new DataManager();
