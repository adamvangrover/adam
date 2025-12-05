// services/webapp/client/src/utils/DataManager.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5001/api';

const ENDPOINT_MAPPING = {
  '/market/data': '/data/adam_market_baseline.json',
  '/graph/data': '/data/knowledge_graph.json',
  '/agents/status': '/data/agents_status.json',
  '/system/health': '/data/system_health.json',
  '/news/feed': '/data/news_feed.json',
  '/knowledge-graph/stats': '/data/system_health.json' // Mock mapping
};

class DataManager {
  constructor() {
    this.useFallback = false;
    this.connectionChecked = false;
  }

  async checkConnection() {
    if (this.connectionChecked) return;

    try {
      // Simple health check endpoint. If it fails, we assume backend is down.
      await axios.get(`${API_BASE_URL}/health`, { timeout: 2000 });
      this.useFallback = false;
      console.log('Connected to Backend API');
    } catch (error) {
      this.useFallback = true;
      console.log('Backend unreachable, switching to Static Simulation Mode');
    } finally {
        this.connectionChecked = true;
    }
  }

  async getData(endpoint) {
    await this.checkConnection();

    if (this.useFallback) {
        return this.loadLocalJson(endpoint);
    }

    try {
      const response = await axios.get(`${API_BASE_URL}${endpoint}`);
      return response.data;
    } catch (error) {
      console.warn(`API call failed for ${endpoint}, falling back to local data.`);
      // If a specific call fails, we might just want to fallback for this call
      // or switch to fallback mode entirely. For robustness, let's switch entirely for now.
      this.useFallback = true;
      return this.loadLocalJson(endpoint);
    }
  }

  async sendAdaptiveQuery(query, context = {}) {
    await this.checkConnection();

    if (this.useFallback) {
      console.log("Offline Mode: Simulating Adaptive Query");
      return {
        status: "Offline Simulation",
        data: {
             human_readable_status: "Simulating Deep Dive for " + query + " (Offline Mode)...",
             v23_knowledge_graph: { nodes: {} }
        }
      };
    }

    try {
      const response = await axios.post(`${API_BASE_URL}/adaptive/query`, {
        query,
        context
      });
      return response.data;
    } catch (error) {
      console.error("Adaptive Query failed:", error);
      throw error;
    }
  }

  async loadLocalJson(endpoint) {
    let localPath = ENDPOINT_MAPPING[endpoint];

    // Fuzzy matching for flexible endpoints
    if (!localPath) {
        if (endpoint.includes('market')) localPath = ENDPOINT_MAPPING['/market/data'];
        else if (endpoint.includes('graph')) localPath = ENDPOINT_MAPPING['/graph/data'];
        else if (endpoint.includes('agents')) localPath = ENDPOINT_MAPPING['/agents/status'];
        else if (endpoint.includes('news')) localPath = ENDPOINT_MAPPING['/news/feed'];
        else {
             console.error(`No local mapping found for ${endpoint}`);
             return null;
        }
    }

    try {
      // fetch is native, relative paths work for files in public/
      const response = await fetch(localPath);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`Failed to load local data from ${localPath}:`, error);
      return null;
    }
  }

  isOfflineMode() {
      return this.useFallback;
  }
}

export const dataManager = new DataManager();
