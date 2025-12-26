import React, { useState, useEffect } from 'react';
import GlassCard from '../common/GlassCard';
import { dataManager } from '../../utils/DataManager';
import { Filter, ArrowUp, ArrowDown, X } from 'lucide-react';

const AgentRegistry = () => {
    const [agents, setAgents] = useState([]);
    const [sortConfig, setSortConfig] = useState({ key: null, direction: 'ascending' });
    const [filter, setFilter] = useState('');

    useEffect(() => {
        const fetchAgents = async () => {
            const data = await dataManager.getData('/agents/status');
            if (data) setAgents(data);
        };
        fetchAgents();
    }, []);

    const requestSort = (key) => {
        let direction = 'ascending';
        if (sortConfig.key === key && sortConfig.direction === 'ascending') {
            direction = 'descending';
        }
        setSortConfig({ key, direction });
    };

    const sortedAgents = React.useMemo(() => {
        let sortableAgents = [...agents];
        if (sortConfig.key) {
            sortableAgents.sort((a, b) => {
                if (a[sortConfig.key] < b[sortConfig.key]) {
                    return sortConfig.direction === 'ascending' ? -1 : 1;
                }
                if (a[sortConfig.key] > b[sortConfig.key]) {
                    return sortConfig.direction === 'ascending' ? 1 : -1;
                }
                return 0;
            });
        }
        return sortableAgents;
    }, [agents, sortConfig]);

    const filteredAgents = sortedAgents.filter(agent =>
        agent.name.toLowerCase().includes(filter.toLowerCase()) ||
        agent.task.toLowerCase().includes(filter.toLowerCase())
    );

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h1 className="text-2xl font-mono text-cyan-400 font-bold">Agent Registry</h1>
                <div className="relative">
                    <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-500" size={16} />
                    <input
                        type="text"
                        aria-label="Filter agents"
                        placeholder="Filter agents..."
                        className="bg-slate-900 border border-slate-700 rounded-md py-2 pl-10 pr-10 text-slate-200 focus:outline-none focus:border-cyan-500 transition-colors"
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                    />
                    {filter && (
                        <button
                            onClick={() => setFilter('')}
                            className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-500 hover:text-white focus:outline-none focus:text-cyan-400 transition-colors"
                            aria-label="Clear filter"
                        >
                            <X size={16} />
                        </button>
                    )}
                </div>
            </div>

            <GlassCard className="overflow-hidden p-0">
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="bg-slate-900/50 text-slate-400 text-sm uppercase tracking-wider border-b border-slate-700">
                            <th className="p-4 cursor-pointer hover:text-white focus:outline-none focus:text-cyan-400"
                                onClick={() => requestSort('name')}
                                tabIndex={0}
                                onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && requestSort('name')}
                                role="columnheader"
                                aria-sort={sortConfig.key === 'name' ? sortConfig.direction : 'none'}
                            >
                                <div className="flex items-center">Agent Name {sortConfig.key === 'name' && (sortConfig.direction === 'ascending' ? <ArrowUp size={14} className="ml-1"/> : <ArrowDown size={14} className="ml-1"/>)}</div>
                            </th>
                            <th className="p-4 cursor-pointer hover:text-white focus:outline-none focus:text-cyan-400"
                                onClick={() => requestSort('status')}
                                tabIndex={0}
                                onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && requestSort('status')}
                                role="columnheader"
                                aria-sort={sortConfig.key === 'status' ? sortConfig.direction : 'none'}
                            >
                                <div className="flex items-center">Status {sortConfig.key === 'status' && (sortConfig.direction === 'ascending' ? <ArrowUp size={14} className="ml-1"/> : <ArrowDown size={14} className="ml-1"/>)}</div>
                            </th>
                            <th className="p-4 cursor-pointer hover:text-white focus:outline-none focus:text-cyan-400"
                                onClick={() => requestSort('task')}
                                tabIndex={0}
                                onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && requestSort('task')}
                                role="columnheader"
                                aria-sort={sortConfig.key === 'task' ? sortConfig.direction : 'none'}
                            >
                                <div className="flex items-center">Current Task {sortConfig.key === 'task' && (sortConfig.direction === 'ascending' ? <ArrowUp size={14} className="ml-1"/> : <ArrowDown size={14} className="ml-1"/>)}</div>
                            </th>
                            <th className="p-4 cursor-pointer hover:text-white focus:outline-none focus:text-cyan-400"
                                onClick={() => requestSort('efficiency')}
                                tabIndex={0}
                                onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && requestSort('efficiency')}
                                role="columnheader"
                                aria-sort={sortConfig.key === 'efficiency' ? sortConfig.direction : 'none'}
                            >
                                <div className="flex items-center">Efficiency {sortConfig.key === 'efficiency' && (sortConfig.direction === 'ascending' ? <ArrowUp size={14} className="ml-1"/> : <ArrowDown size={14} className="ml-1"/>)}</div>
                            </th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                        {filteredAgents.map((agent) => (
                            <tr key={agent.id} className="hover:bg-slate-800/30 transition-colors">
                                <td className="p-4 font-medium text-slate-200">{agent.name}</td>
                                <td className="p-4">
                                    <span className={`px-2 py-1 rounded-full text-xs font-semibold
                                        ${agent.status === 'active' ? 'bg-emerald-900/50 text-emerald-400 border border-emerald-700/50' :
                                          agent.status === 'processing' ? 'bg-cyan-900/50 text-cyan-400 border border-cyan-700/50' :
                                          'bg-slate-700 text-slate-400'}`}>
                                        {agent.status}
                                    </span>
                                </td>
                                <td className="p-4 text-slate-400 font-mono text-xs">{agent.task}</td>
                                <td className="p-4">
                                    <div className="flex items-center space-x-2">
                                        <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden w-24">
                                            <div
                                                className={`h-full rounded-full ${agent.efficiency > 90 ? 'bg-emerald-500' : agent.efficiency > 70 ? 'bg-amber-500' : 'bg-rose-500'}`}
                                                style={{ width: `${agent.efficiency}%` }}
                                            ></div>
                                        </div>
                                        <span className="text-xs text-slate-500">{agent.efficiency}%</span>
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
                {filteredAgents.length === 0 && (
                    <div className="p-8 text-center text-slate-500">No agents found matching filter.</div>
                )}
            </GlassCard>
        </div>
    );
};

export default AgentRegistry;
