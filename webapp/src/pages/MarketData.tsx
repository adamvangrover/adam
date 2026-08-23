import React, { useMemo, Suspense, lazy } from 'react';
import Tabs from '../components/market-data/Tabs';

// ⚡ Bolt: Added code splitting for market data tabs.
// These components likely fetch data or contain complex charts. Loading them
// all upfront blocks the main thread and increases initial bundle size.
// Now, only the active tab's chunk is loaded over the network.
const Stocks = lazy(() => import('../components/market-data/Stocks'));
const Bonds = lazy(() => import('../components/market-data/Bonds'));
const ETFs = lazy(() => import('../components/market-data/ETFs'));
const Crypto = lazy(() => import('../components/market-data/Crypto'));

const LoadingFallback = () => (
  <div className="p-8 text-center text-cyber-cyan font-mono animate-pulse">
    LOADING DATA MODULE...
  </div>
);

const MarketData: React.FC = () => {
  // Bolt Optimization: Memoize tabs configuration to prevent re-mounting/re-rendering
  // of content components (Stocks, Bonds, etc.) on every MarketData render.
  const tabs = useMemo(() => [
    { label: 'Stocks', content: <Suspense fallback={<LoadingFallback />}><Stocks /></Suspense> },
    { label: 'Bonds', content: <Suspense fallback={<LoadingFallback />}><Bonds /></Suspense> },
    { label: 'ETFs', content: <Suspense fallback={<LoadingFallback />}><ETFs /></Suspense> },
    { label: 'Crypto', content: <Suspense fallback={<LoadingFallback />}><Crypto /></Suspense> },
  ], []);

  return (
    <div>
      <h1>Market Data</h1>
      <Tabs tabs={tabs} />
    </div>
  );
};

export default MarketData;