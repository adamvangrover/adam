import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Network, Newspaper, Users, Activity, Settings, LogOut } from 'lucide-react';
import { useTranslation } from 'react-i18next';

const Sidebar = ({ isCollapsed, toggleSidebar, onLogout }) => {
  const { t } = useTranslation();

  const navItems = [
    { to: '/', icon: LayoutDashboard, label: 'Mission Control' }, // Was dashboard.title
    { to: '/knowledge-graph', icon: Network, label: 'Neural Graph' }, // Was knowledgeGraph.title
    { to: '/market-data', icon: Activity, label: 'Market Data' }, // Was marketData.title
    { to: '/agents', icon: Users, label: 'Agent Registry' }, // New
    { to: '/intel', icon: Newspaper, label: 'Intel Feed' }, // New
    { to: '/simulations', icon: Settings, label: 'Simulations' }, // Was simulations.title
  ];

  return (
    <aside className={`h-screen bg-slate-900 border-r border-slate-800 transition-all duration-300 flex flex-col ${isCollapsed ? 'w-20' : 'w-64'}`}>
      <div className="h-16 flex items-center justify-center border-b border-slate-800">
        <h1 className={`font-mono font-bold text-cyan-500 text-xl tracking-wider transition-opacity ${isCollapsed ? 'opacity-0 hidden' : 'opacity-100'}`}>
          ADAM v23.5
        </h1>
        {isCollapsed && <span className="text-cyan-500 font-bold">A</span>}
      </div>

      <nav className="flex-1 py-6 space-y-2 px-3">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `flex items-center px-3 py-3 rounded-md transition-colors duration-200 group ${
                isActive
                  ? 'bg-cyan-900/20 text-cyan-400 border-l-2 border-cyan-400'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-100'
              }`
            }
          >
            <item.icon size={20} className={`${isCollapsed ? 'mx-auto' : 'mr-3'}`} />
            {!isCollapsed && <span className="font-medium">{item.label}</span>}

            {/* Tooltip for collapsed state */}
            {isCollapsed && (
               <div className="absolute left-16 bg-slate-800 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-none whitespace-nowrap">
                  {item.label}
               </div>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="p-4 border-t border-slate-800">
         <button onClick={onLogout} className="flex items-center w-full px-3 py-2 text-rose-400 hover:bg-rose-900/20 rounded-md transition-colors">
            <LogOut size={20} className={`${isCollapsed ? 'mx-auto' : 'mr-3'}`} />
            {!isCollapsed && <span>{t('app.logout') || 'Logout'}</span>}
         </button>
      </div>
    </aside>
  );
};

export default Sidebar;
