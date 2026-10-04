import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorStateProps {
  title?: string;
  message?: string;
  onRetry?: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'BACKEND OFFLINE',
  message = 'Unable to communicate with the TransitVision AI backend service. Verify that the FastAPI server is running on port 8000.',
  onRetry,
}) => {
  return (
    <div className="state-container">
      <AlertTriangle size={40} style={{ color: 'var(--color-danger)' }} />
      <div className="state-title">{title}</div>
      <div className="state-desc">{message}</div>
      {onRetry && (
        <button className="btn btn-secondary btn-sm" onClick={onRetry} style={{ marginTop: '8px' }}>
          <RefreshCw size={14} />
          Retry Connection
        </button>
      )}
    </div>
  );
};
