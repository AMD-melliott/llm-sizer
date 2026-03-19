import { useRef, useEffect, useState } from 'react';
import { useLogStream } from './hooks/useLogStream';

interface LogViewerProps {
  containerId: string;
  apiBase: string;
}

export function LogViewer({ containerId, apiBase }: LogViewerProps) {
  const { lines, connected, error, droppedCount, clear } = useLogStream({
    containerId,
    apiBase,
  });
  const [paused, setPaused] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!paused && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines, paused]);

  return (
    <div className="bg-gray-950 border border-gray-700 rounded">
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-gray-700 text-xs">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-gray-400">{connected ? 'Connected' : 'Disconnected'}</span>
          {droppedCount > 0 && (
            <span className="text-yellow-400">{droppedCount} lines dropped</span>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setPaused(!paused)}
            className="text-gray-400 hover:text-white"
          >
            {paused ? 'Resume' : 'Pause'}
          </button>
          <button
            onClick={clear}
            className="text-gray-400 hover:text-white"
          >
            Clear
          </button>
        </div>
      </div>

      <div
        ref={scrollRef}
        className="p-2 font-mono text-xs max-h-64 overflow-y-auto"
      >
        {error && <div className="text-red-400 mb-1">{error}</div>}
        {lines.length === 0 && !error && (
          <div className="text-gray-600">Waiting for logs...</div>
        )}
        {lines.map((line, i) => (
          <div
            key={i}
            className={line.stream === 'stderr' ? 'text-red-400' : 'text-green-300'}
          >
            <span className="text-gray-600 mr-2 select-none">
              {new Date(line.timestamp).toLocaleTimeString()}
            </span>
            {line.line}
          </div>
        ))}
      </div>
    </div>
  );
}
