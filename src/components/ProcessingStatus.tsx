/**
 * Processing Status Component
 * Shows the current status of file processing
 */

import type { ProcessingStatus } from '../types';

interface ProcessingStatusProps {
  status: ProcessingStatus;
  progress: number;
  message: string;
  error?: Error | null;
}

export function ProcessingStatusDisplay({
  status,
  progress,
  message,
  error,
}: ProcessingStatusProps) {
  if (status === 'idle') {
    return null;
  }

  return (
    <div className={`processing-status status-${status}`}>
      {status === 'error' ? (
        <div className="error-content">
          <svg
            className="error-icon"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
          <span>{error?.message || message}</span>
        </div>
      ) : status === 'complete' ? (
        <div className="success-content">
          <svg
            className="success-icon"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
          </svg>
          <span>{message}</span>
        </div>
      ) : (
        <div className="loading-content">
          <div className="spinner" />
          <div className="progress-info">
            <span>{message}</span>
            {progress > 0 && progress < 100 && (
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${progress}%` }}
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
