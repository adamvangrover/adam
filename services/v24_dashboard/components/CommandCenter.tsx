import React from 'react';

interface Portal {
  title: string;
  path: string;
  icon: string; // Emoji or Lucide icon name (simulated with text for now)
  description: string;
  color: string;
}

const portals: Portal[] = [
  {
    title: "System Knowledge Graph",
    path: "/system_knowledge_graph.html",
    icon: "🕸️",
    description: "Interactive visualization of the entire codebase and agent network.",
    color: "border-purple-500 text-purple-400"
  },
  {
    title: "War Room",
    path: "/war_room_v2.html",
    icon: "🛡️",
    description: "Real-time tactical monitoring and risk assessment dashboard.",
    color: "border-red-500 text-red-400"
  },
  {
    title: "Glitch Monitor",
    path: "/macro_glitch_monitor.html",
    icon: "👁️",
    description: "Detecting anomalies and systemic glitches in macro-economic data.",
    color: "border-cyan-500 text-cyan-400"
  },
  {
    title: "Terminal Access",
    path: "/terminal.html",
    icon: "💻",
    description: "Direct command line interface to the Adam v30 core.",
    color: "border-green-500 text-green-400"
  },
  {
    title: "Market Mayhem",
    path: "/market_mayhem.html",
    icon: "📉",
    description: "Historical and simulated market crash scenarios.",
    color: "border-yellow-500 text-yellow-400"
  },
  {
    title: "Crisis Simulator",
    path: "/crisis_simulator.html",
    icon: "🔥",
    description: "Run Monte Carlo simulations on potential crisis events.",
    color: "border-orange-500 text-orange-400"
  }
];

const CommandCenter: React.FC = () => {
  return (
    <div className="w-full p-6 bg-black/90 rounded-xl border border-gray-800 shadow-2xl">
      <h2 className="text-xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600 font-mono tracking-widest uppercase">
        <span className="mr-2">⚡</span> Command Center Protocols
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {portals.map((portal) => (
          <a
            key={portal.path}
            href={portal.path}
            target="_blank"
            rel="noopener noreferrer"
            className={`group relative p-6 bg-gray-900/50 rounded-lg border ${portal.color} border-opacity-30 hover:border-opacity-100 transition-all duration-300 hover:shadow-[0_0_15px_rgba(0,0,0,0.5)] hover:-translate-y-1 block`}
          >
            {/* Scanline effect overlay */}
            <div className="absolute inset-0 bg-gradient-to-b from-transparent via-white/5 to-transparent opacity-0 group-hover:opacity-20 pointer-events-none transition-opacity duration-500" />

            <div className="flex items-center justify-between mb-3">
              <span className="text-3xl filter drop-shadow-lg">{portal.icon}</span>
              <span className="text-[10px] font-mono uppercase tracking-widest text-gray-500 group-hover:text-white transition-colors">
                SECURE LINK
              </span>
            </div>

            <h3 className={`text-lg font-bold font-mono mb-2 ${portal.color} brightness-110 group-hover:brightness-150 transition-all`}>
              {portal.title}
            </h3>

            <p className="text-xs text-gray-400 font-mono leading-relaxed group-hover:text-gray-300">
              {portal.description}
            </p>

            <div className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
               <span className="text-[10px] font-mono text-white bg-gray-800 px-2 py-0.5 rounded border border-gray-600">ENTER &gt;&gt;</span>
            </div>
          </a>
        ))}
      </div>
    </div>
  );
};

export default CommandCenter;
