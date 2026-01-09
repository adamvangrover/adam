import React from 'react';

const Loading: React.FC = () => {
  return (
    <div
      role="status"
      aria-live="polite"
      aria-label="Loading content"
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100%',
        color: '#00f3ff',
        fontFamily: 'monospace',
        fontSize: '1.2rem',
        flexDirection: 'column'
      }}>
      {/* Bolt Optimization: Replaced inline style injection with CSS class to prevent layout thrashing */}
      <div className="loading-spinner" aria-hidden="true"></div>
      <div>LOADING_MODULE...</div>
    </div>
  );
};

export default Loading;
