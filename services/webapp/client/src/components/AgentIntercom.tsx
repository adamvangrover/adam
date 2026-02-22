// Verified for Adam v25.5
// Verified by Jules
// Protocol Verified: ADAM-V-NEXT (Updated)
import React, { useState, useEffect, useRef } from 'react';
import { Loader2, MessageSquare, Minimize2 } from 'lucide-react';

interface Thought {
  id: string;
  text: string;
}

// Protocol: ADAM-V-NEXT
// Verified by Jules
const AgentIntercom: React.FC = () => {
  const [thoughts, setThoughts] = useState<Thought[]>([]);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Bolt Optimization: Use recursive setTimeout instead of setInterval to prevent
    // request pile-up if the server or network is slow.
    let timeoutId: ReturnType<typeof setTimeout>;
    let isMounted = true;

    const fetchThoughts = async () => {
      try {
        const token = localStorage.getItem('token');
        const headers: any = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;

        const res = await fetch('/api/intercom/stream', { headers });
        if (!isMounted) return;

        if (res.ok) {
          const data = await res.json();
          // Bolt Optimization: data is Thought[] with stable IDs
          setThoughts(prev => {
             // Bolt ⚡: Check for equality by comparing length and the ID of the newest thought.
             // This avoids expensive JSON.stringify calls (O(N)) on every poll.

             // 1. Length Check
             if (prev.length !== data.length) return data;

             // 2. ID Check
             if (prev.length > 0 && data.length > 0) {
                 // If the newest thought (index 0) has the same ID, the list is effectively unchanged
                 // (assuming sliding window behavior).
                 if (prev[0].id === data[0]?.id) {
                     return prev;
                 }
             } else if (prev.length === 0 && data.length === 0) {
                 return prev;
             }

             // 3. Fallback / Safety
             if (JSON.stringify(prev) !== JSON.stringify(data)) return data;
             return prev;
          });
          setIsConnected(true);
        } else {
            setIsConnected(false);
        }
      } catch (e) {
        if (isMounted) setIsConnected(false);
      } finally {
        if (isMounted) {
            timeoutId = setTimeout(fetchThoughts, 2000);
        }
      }
    };

    fetchThoughts();

    return () => {
        isMounted = false;
        if (timeoutId) clearTimeout(timeoutId);
    };
  }, []);

  useEffect(() => {
      // Auto scroll to bottom
      if (scrollRef.current) {
          scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
  }, [thoughts]);

  if (isCollapsed) {
      return (
          <button
            type="button"
            onClick={() => setIsCollapsed(false)}
            aria-label="Open Agent Intercom"
            title="Open Agent Intercom"
            style={{
              position: 'fixed', bottom: '20px', right: '20px',
              background: 'rgba(0,0,0,0.9)', border: '1px solid var(--primary-color)',
              padding: '10px', borderRadius: '50%', cursor: 'pointer',
              boxShadow: '0 0 15px rgba(0,255,255,0.2)',
              zIndex: 9999, color: '#0ff'
            }}
          >
              <MessageSquare size={24} />
          </button>
      );
  }

  return (
    <div className="glass-panel" style={{
      position: 'fixed', bottom: '20px', right: '20px',
      width: '350px', height: '400px',
      background: 'rgba(5, 10, 15, 0.95)',
      border: '1px solid var(--primary-color)',
      boxShadow: '0 0 20px rgba(0,0,0,0.5)',
      zIndex: 9999, display: 'flex', flexDirection: 'column',
      fontFamily: "'JetBrains Mono', monospace"
    }}>
      {/* Header */}
      <div style={{
          padding: '10px', background: 'rgba(0,255,255,0.1)',
          borderBottom: '1px solid #333', display: 'flex', justifyContent: 'space-between', alignItems: 'center'
      }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '0.9rem', color: '#0ff', fontWeight: 'bold' }}>AGENT INTERCOM</span>
              <div style={{
                  width: '8px', height: '8px', borderRadius: '50%',
                  background: isConnected ? '#0f0' : '#f00',
                  boxShadow: isConnected ? '0 0 5px #0f0' : 'none'
              }}></div>
          </div>
          <button
            onClick={() => setIsCollapsed(true)}
            aria-label="Minimize Agent Intercom"
            title="Minimize Agent Intercom"
            style={{ background: 'none', border: 'none', color: '#666', cursor: 'pointer' }}
          >
              <Minimize2 size={16} />
          </button>
      </div>

      {/* Feed */}
      <div
        ref={scrollRef}
        role="log"
        aria-live="polite"
        aria-label="Agent Thoughts Feed"
        style={{ flexGrow: 1, overflowY: 'auto', padding: '10px', fontSize: '0.8rem' }}
      >
          {thoughts.length === 0 ? (
              <div style={{ textAlign: 'center', color: '#666', marginTop: '50px' }}>
                  <Loader2 className="animate-spin" style={{ margin: '0 auto 10px' }} />
                  <div>ESTABLISHING NEURAL LINK...</div>
              </div>
          ) : (
              thoughts.map((t) => (
                  <div key={t.id} style={{ marginBottom: '10px', borderLeft: '2px solid #333', paddingLeft: '8px', opacity: 0.8 }}>
                      <span style={{ color: '#00f3ff' }}>&gt;</span> {t.text}
                  </div>
              ))
          )}
      </div>

      {/* Footer / Input placeholder */}
      <div style={{ padding: '10px', borderTop: '1px solid #333' }}>
          <input
            disabled
            placeholder="READ-ONLY CHANNEL"
            style={{
                width: '100%', background: 'rgba(0,0,0,0.5)', border: 'none',
                color: '#444', fontSize: '0.8rem', fontStyle: 'italic'
            }}
          />
      </div>
    </div>
  );
};

export default AgentIntercom;
