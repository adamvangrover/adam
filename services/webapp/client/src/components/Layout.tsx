import React from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import GlobalNav from './GlobalNav';

const Layout: React.FC = () => {
  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <GlobalNav />
      <div style={{ display: 'flex', flexGrow: 1, overflow: 'hidden' }}>
        <Sidebar />
        <main style={{ flexGrow: 1, padding: '20px', overflowY: 'auto', backgroundColor: 'var(--bg-color)' }}>
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default Layout;
