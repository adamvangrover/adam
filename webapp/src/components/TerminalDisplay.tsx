import React, { useEffect, useRef } from 'react';

export interface TerminalLineItem {
  id: string;
  text: string;
}

interface TerminalDisplayProps {
  lines: TerminalLineItem[];
}

// ⚡ Bolt: Extract and memoize terminal line to prevent O(N) re-renders
// when new items are appended to the sliding window list.
const TerminalLine = React.memo(({ line }: { line: string }) => (
  <div className="break-all">{line}</div>
));
TerminalLine.displayName = 'TerminalLine';

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
        {lines.map((line) => (
            // ⚡ Bolt: Assign a stable ID key instead of index to prevent O(N) unmounts/remounts
            // when the list behaves as a sliding window.
            <TerminalLine key={line.id} line={line.text} />
        ))}
    </div>
  );
};

export default React.memo(TerminalDisplay);
