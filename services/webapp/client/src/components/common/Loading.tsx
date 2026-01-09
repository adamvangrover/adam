import React from 'react';

const Loading: React.FC = () => {
  return (
    <div
      className="flex flex-col justify-center items-center h-full text-[#00f3ff] font-mono text-lg"
      role="status"
      aria-live="polite"
      aria-label="Loading module content"
    >
      <div className="loading-spinner"></div>
      <div>LOADING_MODULE...</div>
    </div>
  );
};

export default Loading;
