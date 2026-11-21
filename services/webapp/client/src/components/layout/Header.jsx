import React from 'react';
import { Menu, Wifi, WifiOff, Bell } from 'lucide-react';

const Header = ({ toggleSidebar, isOffline }) => {
  return (
    <header className="h-16 bg-slate-900/50 backdrop-blur-sm border-b border-slate-800 flex items-center justify-between px-6 sticky top-0 z-10">
      <div className="flex items-center">
        <button onClick={toggleSidebar} className="text-slate-400 hover:text-white p-2 rounded-md hover:bg-slate-800 transition-colors">
          <Menu size={24} />
        </button>
      </div>

      <div className="flex items-center space-x-6">
        {/* System Status Indicator */}
        <div className={`flex items-center space-x-2 px-3 py-1 rounded-full border ${isOffline ? 'bg-amber-900/20 border-amber-700/50 text-amber-500' : 'bg-emerald-900/20 border-emerald-700/50 text-emerald-500'}`}>
            {isOffline ? <WifiOff size={16} /> : <Wifi size={16} />}
            <span className="text-xs font-mono uppercase tracking-wide font-semibold">
                {isOffline ? 'Static Simulation Mode' : 'Connected to Core'}
            </span>
        </div>

        <button className="relative text-slate-400 hover:text-white transition-colors">
            <Bell size={20} />
            <span className="absolute -top-1 -right-1 w-2 h-2 bg-rose-500 rounded-full animate-pulse"></span>
        </button>

        <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-cyan-500 to-blue-600 border border-slate-600 shadow-lg shadow-cyan-500/20"></div>
      </div>
    </header>
  );
};

export default Header;
