import React from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import GlobalNav from './GlobalNav';
import AgentIntercom from './AgentIntercom';

const Layout: React.FC = () => {
  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <GlobalNav />
      <div style={{ display: 'flex', flexGrow: 1, overflow: 'hidden' }}>
        <Sidebar />
        <main style={{ flexGrow: 1, padding: '30px 40px', overflowY: 'auto', backgroundColor: 'var(--bg-color)' }} className="scrollbar-hide">
          <Outlet />
        </main>
        <AgentIntercom />
      </div>
    </div>
  );
};

export default Layout;
