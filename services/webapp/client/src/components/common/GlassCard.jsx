import React from 'react';

const GlassCard = ({ children, className = '' }) => {
  return (
    <div className={`glass-panel rounded-lg p-6 border border-slate-700/50 ${className}`}>
      {children}
    </div>
  );
};

export default GlassCard;
