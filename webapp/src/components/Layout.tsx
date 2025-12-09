import React from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import GlobalNav from './GlobalNav';

// Placeholder for data context (will be implemented in Phase 2)
const isOffline = true; // Default to simulation for now

const Layout: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col bg-cyber-black text-cyber-text font-sans">
      <div className="scan-line"></div>

      <GlobalNav isOffline={isOffline} />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar />

        <main className="flex-1 overflow-y-auto p-6 relative">
          <div className="max-w-7xl mx-auto space-y-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};

export default Layout;
