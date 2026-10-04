import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({ message = 'Loading real-time telemetry...' }) => {
  return (
    <div className="state-container">
      <Loader2 size={32} style={{ color: 'var(--color-primary)', animation: 'spin 1.2s linear infinite' }} />
      <div className="state-title">{message}</div>
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};
