import React, { useMemo } from 'react';
import Tabs from '../components/market-data/Tabs';
import Stocks from '../components/market-data/Stocks';
import Bonds from '../components/market-data/Bonds';
import ETFs from '../components/market-data/ETFs';
import Crypto from '../components/market-data/Crypto';

const MarketData: React.FC = () => {
  // Bolt Optimization: Memoize tabs configuration to prevent re-mounting/re-rendering
  // of content components (Stocks, Bonds, etc.) on every MarketData render.
  const tabs = useMemo(() => [
    { label: 'Stocks', content: <Stocks /> },
    { label: 'Bonds', content: <Bonds /> },
    { label: 'ETFs', content: <ETFs /> },
    { label: 'Crypto', content: <Crypto /> },
  ], []);

  return (
    <div>
      <h1>Market Data</h1>
      <Tabs tabs={tabs} />
    </div>
  );
};

export default MarketData;
