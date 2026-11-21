import React, { useState } from 'react';
import Sidebar from './Sidebar';
import Header from './Header';

const Layout = ({ children, onLogout, isOffline }) => {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 overflow-hidden font-sans bg-grid-pattern relative">
      <div className="absolute inset-0 bg-slate-950/90 pointer-events-none"></div>
      <div className="absolute inset-0 scanline z-50 pointer-events-none opacity-50"></div>

      <Sidebar
        isCollapsed={isSidebarCollapsed}
        toggleSidebar={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
        onLogout={onLogout}
      />

      <div className="flex-1 flex flex-col relative z-0 overflow-hidden">
        <Header
            toggleSidebar={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
            isOffline={isOffline}
        />
        <main className="flex-1 overflow-y-auto p-6 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout;
