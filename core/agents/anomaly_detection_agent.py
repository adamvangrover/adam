# core/agents/anomaly_detection_agent.py

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Union
import logging
import random
from core.engine.unified_knowledge_graph import UnifiedKnowledgeGraph
from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class AnomalyDetectionAgent(AgentBase):
    """
    Detects anomalies and unusual patterns in financial markets and company data.

    Core Capabilities:
    - Leverages various statistical methods and machine learning algorithms for comprehensive anomaly detection.
    - Integrates with Adam's knowledge base for context-aware analysis.
    - Employs XAI techniques to provide explanations for detected anomalies.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initializes the AnomalyDetectionAgent with configuration parameters.
        """
        super().__init__(config, **kwargs)
        # Load initially, but can be overridden in execute
        self.market_data = self._load_market_data()
        self.company_data = self._load_company_data(self.config.get("ticker", "AAPL"))
        self.scaler = StandardScaler()

    async def execute(self, input_data: Union[str, AgentInput, Dict[str, Any]] = None, **kwargs) -> Union[Dict[str, Any], AgentOutput]:
        """
        Executes anomaly detection based on standard inputs.
        """
        is_standard_mode = False
        query = "Anomaly Detection"
        ticker = self.config.get("ticker", "AAPL")
        target_data_type = "both" # "market", "company", "both"

        if input_data is not None:
            if isinstance(input_data, AgentInput):
                query = input_data.query
                ticker = input_data.context.get("ticker", ticker)
                target_data_type = input_data.context.get("target_data_type", "both")
                # Override internal data if provided
                if "market_data" in input_data.context:
                    self.market_data = pd.DataFrame(input_data.context["market_data"])
                if "company_data" in input_data.context:
                    self.company_data = pd.DataFrame(input_data.context["company_data"])
                is_standard_mode = True
            elif isinstance(input_data, dict):
                ticker = input_data.get("ticker", ticker)
                target_data_type = input_data.get("target_data_type", "both")
                if "market_data" in input_data:
                    self.market_data = pd.DataFrame(input_data["market_data"])
                if "company_data" in input_data:
                    self.company_data = pd.DataFrame(input_data["company_data"])
                kwargs.update(input_data)
            elif isinstance(input_data, str):
                query = input_data
                ticker = input_data # Assuming query might just be a ticker

        # If data wasn't injected, ensure we load for the correct ticker
        if "company_data" not in (input_data.context if isinstance(input_data, AgentInput) else (input_data if isinstance(input_data, dict) else {})):
             self.company_data = self._load_company_data(ticker)

        logger.info(f"Running anomaly detection for {ticker} (Target: {target_data_type})...")

        results = {
            "ticker": ticker,
            "market_anomalies": [],
            "company_anomalies": []
        }

        if target_data_type in ["market", "both"]:
            results["market_anomalies"] = self.detect_market_anomalies()

        if target_data_type in ["company", "both"]:
            results["company_anomalies"] = self.detect_company_anomalies()

        if is_standard_mode:
            answer = f"Anomaly Detection Report for {ticker}:\n"

            if results["market_anomalies"]:
                answer += f"\nMarket Anomalies ({len(results['market_anomalies'])} detected):\n"
                for a in results["market_anomalies"][:5]: # Top 5
                    answer += f"- [{a['method']}] {a['message']}\n"
            else:
                answer += "\nNo significant market anomalies detected.\n"

            if results["company_anomalies"]:
                answer += f"\nCompany Anomalies ({len(results['company_anomalies'])} detected):\n"
                for a in results["company_anomalies"][:5]:
                    answer += f"- [{a['method']}] {a['message']}\n"
            else:
                answer += "\nNo significant company financial anomalies detected.\n"

            total_anomalies = len(results["market_anomalies"]) + len(results["company_anomalies"])

            return AgentOutput(
                answer=answer,
                sources=["Knowledge Graph", "Market Data Models"],
                confidence=0.85 if total_anomalies > 0 else 0.5,
                metadata=results
            )

        return results

    def _load_market_data(self) -> pd.DataFrame:
        """Loads default market data."""
        # Placeholder market data (replace with actual data)
        data = {
            'stock_price': [100, 102, 105, 108, 110, 112, 115, 118, 120, 122, 125, 128, 130, 132, 135, 138, 140, 142, 145, 148, 150, 152, 155, 158, 160, 162, 165, 168, 170, 172, 175, 178, 180, 182, 185, 188, 190, 192, 195, 198, 200, 202, 205, 208, 210, 212, 215, 218, 220, 222, 225, 228, 230, 232, 235, 238, 240, 242, 245, 248, 250],
            'trading_volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000]
        }
        return pd.DataFrame(data)

    def _load_company_data(self, ticker: str) -> pd.DataFrame:
        """Loads company data from Adam's knowledge graph or mock fallback."""
        try:
            kg = UnifiedKnowledgeGraph()

            # Find company node
            company_node = None
            company_data_from_kg = None

            # Search by ticker in node attributes
            for node, attrs in kg.graph.nodes(data=True):
                if attrs.get("ticker") == ticker:
                    company_node = node
                    company_data_from_kg = attrs
                    break

            if company_data_from_kg:
                logger.info(f"Found company data for {ticker} in Knowledge Graph.")
                financials = company_data_from_kg.get("financials", {})

                market_cap = financials.get("market_cap", 1000000000)
                pe_ratio = financials.get("pe_ratio", 20)
                estimated_net_income = market_cap / pe_ratio if pe_ratio else market_cap * 0.05
                estimated_revenue = estimated_net_income / 0.15

                revenues = []
                net_incomes = []
                current_rev = estimated_revenue * 0.6
                current_ni = estimated_net_income * 0.6

                for _ in range(10):
                    growth = random.uniform(0.02, 0.08)
                    noise = random.uniform(0.95, 1.05)
                    current_rev = current_rev * (1 + growth) * noise
                    current_ni = current_rev * 0.15 * random.uniform(0.9, 1.1)

                    revenues.append(current_rev)
                    net_incomes.append(current_ni)

                return pd.DataFrame({'revenue': revenues, 'net_income': net_incomes})

        except Exception as e:
            logger.debug(f"Knowledge Graph not available: {e}. Using placeholder data.")

        # Fallback
        data = {
            'revenue': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000],
            'net_income': [100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000, 180000, 190000],
            'total_assets': [5000000, 5200000, 5500000, 5800000, 6000000, 6200000, 6500000, 6800000, 7000000, 7200000],
            'shareholders_equity': [2000000, 2100000, 2200000, 2300000, 2400000, 2500000, 2600000, 2700000, 2800000, 2900000],
            'total_liabilities': [3000000, 3100000, 3300000, 3500000, 3600000, 3700000, 3900000, 4100000, 4200000, 4300000],
            'current_assets': [1000000, 1050000, 1100000, 1150000, 1200000, 1250000, 1300000, 1350000, 1400000, 1450000],
            'current_liabilities': [500000, 520000, 550000, 580000, 600000, 620000, 650000, 680000, 700000, 720000],
            'inventory': [200000, 210000, 220000, 230000, 240000, 250000, 260000, 270000, 280000, 290000],
            'total_debt': [1500000, 1550000, 1600000, 1650000, 1700000, 1750000, 1800000, 1850000, 1900000, 1950000]
        }
        return pd.DataFrame(data)

    def _detect_outliers_zscore(self, data: pd.Series, threshold: float = 3) -> List[int]:
        if len(data) < 3: return []
        z_scores = np.abs(stats.zscore(data))
        outliers = np.where(z_scores > threshold)[0]
        return outliers.tolist()

    def _detect_outliers_isolation_forest(self, data: pd.DataFrame, contamination: float = 0.1) -> List[int]:
        if len(data) < 10: return []
        model = IsolationForest(contamination=contamination)
        model.fit(data)
        predictions = model.predict(data)
        outliers = np.where(predictions == -1)[0]
        return outliers.tolist()

    def _detect_outliers_lof(self, data: pd.DataFrame, n_neighbors: int = 20) -> List[int]:
        if len(data) <= n_neighbors: return []
        model = LocalOutlierFactor(n_neighbors=n_neighbors)
        predictions = model.fit_predict(data)
        outliers = np.where(predictions == -1)[0]
        return outliers.tolist()

    def _detect_outliers_one_class_svm(self, data: pd.DataFrame, nu: float = 0.1) -> List[int]:
        if len(data) < 5: return []
        model = OneClassSVM(nu=nu)
        model.fit(data)
        predictions = model.predict(data)
        outliers = np.where(predictions == -1)[0]
        return outliers.tolist()

    def _detect_anomalies_clustering(self, data: pd.DataFrame, n_clusters: int = 5) -> List[int]:
        if len(data) <= n_clusters: return []
        # Suppress KMeans warning about memory leak on Windows with MKL
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = KMeans(n_clusters=n_clusters, n_init=10)
            model.fit(data)
            distances = model.transform(data)
            min_distances = np.min(distances, axis=1)
            threshold = np.percentile(min_distances, 95)
            outliers = np.where(min_distances > threshold)[0]
            return outliers.tolist()

    def _detect_anomalies_time_series(self, data: pd.Series) -> List[int]:
        period = 12
        if len(data) < 2 * period:
            return []

        try:
            decomposition = seasonal_decompose(data, model='additive', period=period)
            residual = decomposition.resid.dropna()

            if len(residual) < 10:
                return []

            import warnings
            from statsmodels.tools.sm_exceptions import ConvergenceWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = ARIMA(residual, order=(1, 0, 0)) # Simplified to prevent convergence issues in small samples
                model_fit = model.fit()

            model_residuals = model_fit.resid
            threshold = 3 * np.std(model_residuals)
            outlier_indices = np.where(np.abs(model_residuals) > threshold)[0]
            original_indices = residual.index[outlier_indices].tolist()

            return original_indices
        except Exception as e:
            logger.debug(f"Time series anomaly detection skipped: {e}")
            return []

    def _get_financial_ratios(self, company_data: pd.DataFrame) -> pd.DataFrame:
        ratios = pd.DataFrame(index=company_data.index)

        # 1. Profitability
        if 'net_income' in company_data.columns and 'revenue' in company_data.columns:
            ratios['net_profit_margin'] = company_data['net_income'] / company_data['revenue']
        if 'net_income' in company_data.columns and 'total_assets' in company_data.columns:
            ratios['return_on_assets'] = company_data['net_income'] / company_data['total_assets']
        if 'net_income' in company_data.columns and 'shareholders_equity' in company_data.columns:
            ratios['return_on_equity'] = company_data['net_income'] / company_data['shareholders_equity']

        # 2. Liquidity
        if 'current_assets' in company_data.columns and 'current_liabilities' in company_data.columns:
            ratios['current_ratio'] = company_data['current_assets'] / company_data['current_liabilities']
        if 'current_assets' in company_data.columns and 'inventory' in company_data.columns and 'current_liabilities' in company_data.columns:
            ratios['quick_ratio'] = (company_data['current_assets'] - company_data['inventory']) / company_data['current_liabilities']

        # 3. Solvency
        if 'total_debt' in company_data.columns and 'shareholders_equity' in company_data.columns:
            ratios['debt_to_equity'] = company_data['total_debt'] / company_data['shareholders_equity']
        elif 'total_liabilities' in company_data.columns and 'shareholders_equity' in company_data.columns:
            ratios['debt_to_equity'] = company_data['total_liabilities'] / company_data['shareholders_equity']

        ratios.replace([np.inf, -np.inf], np.nan, inplace=True)
        return ratios

    def detect_market_anomalies(self) -> List[Dict]:
        anomalies = []
        if self.market_data is None or self.market_data.empty:
            return anomalies

        if 'stock_price' in self.market_data.columns:
            outliers = self._detect_outliers_zscore(self.market_data['stock_price'])
            for idx in outliers:
                anomalies.append({
                    'type': 'stock_price_outlier',
                    'value': self.market_data['stock_price'].iloc[idx],
                    'method': 'z-score',
                    'message': f"Unusual stock price movement: {self.market_data['stock_price'].iloc[idx]:.2f}"
                })

        if 'trading_volume' in self.market_data.columns:
            outliers = self._detect_outliers_isolation_forest(self.market_data[['trading_volume']])
            for idx in outliers:
                anomalies.append({
                    'type': 'trading_volume_anomaly',
                    'value': self.market_data['trading_volume'].iloc[idx],
                    'method': 'isolation_forest',
                    'message': f"Unusual trading volume: {self.market_data['trading_volume'].iloc[idx]}"
                })

        # Multivariate if we have both
        if 'stock_price' in self.market_data.columns and 'trading_volume' in self.market_data.columns:
            try:
                scaled_data = self.scaler.fit_transform(self.market_data[['stock_price', 'trading_volume']])

                # LOF
                outliers_lof = self._detect_outliers_lof(pd.DataFrame(scaled_data))
                for idx in outliers_lof:
                    anomalies.append({
                        'type': 'multivariate_anomaly',
                        'method': 'LOF',
                        'message': f"Complex anomaly detected at index {idx} (LOF)."
                    })
            except Exception as e:
                logger.debug(f"Multivariate anomaly detection failed: {e}")

        return anomalies

    def detect_company_anomalies(self) -> List[Dict]:
        anomalies = []
        if self.company_data is None or self.company_data.empty:
            return anomalies

        if 'revenue' in self.company_data.columns:
            outliers = self._detect_outliers_zscore(self.company_data['revenue'], threshold=2)
            for idx in outliers:
                anomalies.append({
                    'type': 'revenue_outlier',
                    'value': self.company_data['revenue'].iloc[idx],
                    'method': 'z-score',
                    'message': f"Unusual revenue value: {self.company_data['revenue'].iloc[idx]:.2f}"
                })

        ratios = self._get_financial_ratios(self.company_data)

        for col in ratios.columns:
            if ratios[col].notna().any():
                outlier_indices = self._detect_outliers_zscore(ratios[col].dropna(), threshold=2.5)
                valid_indices = ratios[col].dropna().index

                for idx in outlier_indices:
                    original_idx = valid_indices[idx]
                    anomalies.append({
                        'type': f'{col}_anomaly',
                        'value': ratios[col].loc[original_idx],
                        'method': 'ratio_z_score',
                        'message': f"Unusual {col}: {ratios[col].loc[original_idx]:.4f}"
                    })

        return anomalies
