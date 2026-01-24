import React, { useEffect, useRef } from 'react';

interface TerminalDisplayProps {
  lines: string[];
}

const TerminalDisplay: React.FC<TerminalDisplayProps> = ({ lines }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines]);

  return (
    <div
        className="flex-1 overflow-y-auto space-y-1 text-green-400 p-2 focus:outline-none focus:ring-1 focus:ring-cyber-cyan/30"
        ref={scrollRef}
        role="log"
        aria-live="polite"
        aria-atomic="false"
        aria-label="Terminal Output"
        tabIndex={0}
    >
        {lines.map((line, i) => (
            <div key={i} className="break-all">{line}</div>
        ))}
    </div>
  );
};

export default React.memo(TerminalDisplay);
