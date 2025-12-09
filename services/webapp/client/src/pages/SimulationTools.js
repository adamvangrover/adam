import React, { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { getAuthHeaders } from '../utils/auth';

const SimulationHistory = ({ history, onSelect }) => {
    const { t } = useTranslation();
    return (
        <div className="Card">
            <h3>{t('simulations.history')}</h3>
            <table>
                <thead>
                    <tr>
                        <th>{t('simulations.name')}</th>
                        <th>{t('simulations.status')}</th>
                        <th>{t('simulations.actions')}</th>
                    </tr>
                </thead>
                <tbody>
                    {history.map(run => (
                        <tr key={run.task_id}>
                            <td>{run.simulation_name}</td>
                            <td>{run.status}</td>
                            <td>
                                <button onClick={() => onSelect(run.task_id)} disabled={run.status !== 'SUCCESS'}>
                                    {t('simulations.viewResult')}
                                </button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

const SimulationResult = ({ result, onClear }) => {
    const { t } = useTranslation();
    return (
        <div className="Card">
            <button onClick={onClear}>{t('simulations.backToSimulations')}</button>
            <h3>{t('simulations.result')}</h3>
            <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
    );
}


function Simulations() {
    const { t } = useTranslation();
    const [simulations, setSimulations] = useState([]);
    const [selectedSimulation, setSelectedSimulation] = useState('');
    const [history, setHistory] = useState([]);
    const [selectedResult, setSelectedResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const fetchSimulations = useCallback(async () => {
        const headers = await getAuthHeaders();
        const response = await fetch('/api/simulations', { headers });
        const data = await response.json();
        setSimulations(data);
        if (data.length > 0) {
            setSelectedSimulation(data[0]);
        }
    }, []);

    const fetchHistory = useCallback(async () => {
        const headers = await getAuthHeaders();
        const response = await fetch('/api/simulations/history', { headers });
        const data = await response.json();
        setHistory(data);
    }, []);

    useEffect(() => {
        fetchSimulations();
        fetchHistory();
    }, [fetchSimulations, fetchHistory]);

    // Polling for status updates
    useEffect(() => {
        const interval = setInterval(() => {
            const running = history.some(run => run.status === 'PENDING' || run.status === 'STARTED');
            if (running) {
                fetchHistory();
            }
        }, 5000);
        return () => clearInterval(interval);
    }, [history, fetchHistory]);


    const handleRunSimulation = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        const headers = await getAuthHeaders();
        await fetch(`/api/simulations/${selectedSimulation}`, {
            method: 'POST',
            headers,
        });
        await fetchHistory(); // Refresh history immediately
        setIsLoading(false);
    };

    const handleViewResult = async (taskId) => {
        setIsLoading(true);
        const headers = await getAuthHeaders();
        const response = await fetch(`/api/tasks/${taskId}`, { headers });
        const data = await response.json();
        setSelectedResult(data.result);
        setIsLoading(false);
    }

    if (isLoading) {
        return <p>{t('analysisTools.loading')}</p>
    }

    if (selectedResult) {
        return <SimulationResult result={selectedResult} onClear={() => setSelectedResult(null)} />;
    }

    return (
        <div>
            <h2>{t('simulations.title')}</h2>
            <div className="Card">
                <h3>{t('simulations.runSimulation')}</h3>
                <form onSubmit={handleRunSimulation}>
                    <select value={selectedSimulation} onChange={e => setSelectedSimulation(e.target.value)}>
                        {simulations.map(sim => (
                            <option key={sim} value={sim}>{sim.replace(/_/g, ' ')}</option>
                        ))}
                    </select>
                    <button type="submit" disabled={!selectedSimulation}>{t('simulations.run')}</button>
                </form>
            </div>

            <SimulationHistory history={history} onSelect={handleViewResult} />
        </div>
    );
}

export default Simulations;
