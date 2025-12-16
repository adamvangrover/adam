import React, { useState } from 'react';

interface LegacyVisualizerProps {
  initialFile?: string;
}

const VISUALIZATIONS = [
  "neural_dashboard.html",
  "deep_dive.html",
  "financial_twin.html",
  "reports.html",
  "dashboard.html",
  "agents.html",
  "graph.html"
];

const LegacyVisualizer: React.FC<LegacyVisualizerProps> = ({ initialFile = "neural_dashboard.html" }) => {
  const [currentFile, setCurrentFile] = useState(initialFile);

  return (
    <div className="flex flex-col h-screen w-full bg-gray-900 text-white">
      <div className="flex p-4 bg-gray-800 border-b border-gray-700">
        <h2 className="text-xl font-bold mr-4">Legacy Visualizations</h2>
        <select
          value={currentFile}
          onChange={(e) => setCurrentFile(e.target.value)}
          className="bg-gray-700 text-white p-2 rounded"
        >
          {VISUALIZATIONS.map(file => (
            <option key={file} value={file}>{file}</option>
          ))}
        </select>
      </div>
      <div className="flex-grow relative">
        <iframe
          src={`/visualizations/${currentFile}`}
          className="w-full h-full border-none"
          title="Legacy Visualization"
        />
      </div>
    </div>
  );
};

export default LegacyVisualizer;
