import React, { useState } from 'react';
import { useContainerStore } from '../../store/useContainerStore';
import type { ValidationLevel } from '../../types';

export const ValidationDisplay: React.FC = () => {
  const { validationResult } = useContainerStore();
  const [isExpanded, setIsExpanded] = useState(true);

  if (!validationResult) {
    return null;
  }

  const { messages, securityIssues, recommendations } = validationResult;

  const getLevelIcon = (level: ValidationLevel) => {
    const colorMap: Record<ValidationLevel, string> = {
      error: 'var(--danger)',
      warning: 'var(--warning)',
      info: 'var(--accent)',
      success: 'var(--success)',
    };
    const color = colorMap[level];

    switch (level) {
      case 'error':
        return (
          <svg className="w-5 h-5" style={{ color }} fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
        );
      case 'warning':
        return (
          <svg className="w-5 h-5" style={{ color }} fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
        );
      case 'info':
        return (
          <svg className="w-5 h-5" style={{ color }} fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
        );
      case 'success':
        return (
          <svg className="w-5 h-5" style={{ color }} fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        );
    }
  };

  const getLevelStyles = (level: ValidationLevel): React.CSSProperties => {
    switch (level) {
      case 'error':
        return { backgroundColor: 'var(--danger-light)', borderColor: 'var(--danger)' };
      case 'warning':
        return { backgroundColor: 'var(--warning-light)', borderColor: 'var(--warning)' };
      case 'info':
        return { backgroundColor: 'var(--accent-light)', borderColor: 'var(--accent)' };
      case 'success':
        return { backgroundColor: 'var(--success-light)', borderColor: 'var(--success)' };
    }
  };

  const getBadgeStyles = (level: 'error' | 'warning' | 'success'): React.CSSProperties => {
    switch (level) {
      case 'error':
        return { backgroundColor: 'var(--danger-light)', color: 'var(--danger-text)' };
      case 'warning':
        return { backgroundColor: 'var(--warning-light)', color: 'var(--warning-text)' };
      case 'success':
        return { backgroundColor: 'var(--success-light)', color: 'var(--success-text)' };
    }
  };

  const hasErrors = messages.some(m => m.level === 'error') || securityIssues.some(m => m.level === 'error');
  const hasWarnings = messages.some(m => m.level === 'warning') || securityIssues.some(m => m.level === 'warning');

  return (
    <div
      className="rounded-xl"
      style={{ backgroundColor: 'var(--surface-inset)', border: '1px solid var(--border)' }}
    >
      {/* Header - Clickable to toggle */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-4 transition-colors rounded-t-xl"
      >
        <div className="flex items-center">
          <svg
            className="w-5 h-5 mr-2"
            style={{ color: 'var(--warning)' }}
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          <h4 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            Validation Status
          </h4>
          {hasErrors && (
            <span
              className="ml-3 px-2 py-1 text-xs font-medium rounded-lg"
              style={getBadgeStyles('error')}
            >
              {messages.filter(m => m.level === 'error').length + securityIssues.filter(m => m.level === 'error').length} Error(s)
            </span>
          )}
          {hasWarnings && !hasErrors && (
            <span
              className="ml-3 px-2 py-1 text-xs font-medium rounded-lg"
              style={getBadgeStyles('warning')}
            >
              {messages.filter(m => m.level === 'warning').length + securityIssues.filter(m => m.level === 'warning').length} Warning(s)
            </span>
          )}
          {!hasErrors && !hasWarnings && (
            <span
              className="ml-3 px-2 py-1 text-xs font-medium rounded-lg"
              style={getBadgeStyles('success')}
            >
              Valid
            </span>
          )}
        </div>
        <svg
          className={`w-5 h-5 transition-transform ${isExpanded ? 'transform rotate-180' : ''}`}
          style={{ color: 'var(--text-muted)' }}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Content - Collapsible */}
      {isExpanded && (
        <div className="p-4 pt-0 space-y-3">
          {/* Overall Status */}
          <div
            className="p-4 rounded-xl"
            style={{
              ...getLevelStyles(hasErrors ? 'error' : hasWarnings ? 'warning' : 'success'),
              borderWidth: '1px',
              borderStyle: 'solid',
            }}
          >
            <div className="flex items-center">
              {hasErrors ? (
                <>
                  <svg className="w-6 h-6 mr-2" style={{ color: 'var(--danger)' }} fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                  <span className="font-medium" style={{ color: 'var(--danger-text)' }}>
                    Configuration has errors
                  </span>
                </>
              ) : hasWarnings ? (
                <>
                  <svg className="w-6 h-6 mr-2" style={{ color: 'var(--warning)' }} fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  <span className="font-medium" style={{ color: 'var(--warning-text)' }}>
                    Configuration valid with warnings
                  </span>
                </>
              ) : (
                <>
                  <svg className="w-6 h-6 mr-2" style={{ color: 'var(--success)' }} fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  <span className="font-medium" style={{ color: 'var(--success-text)' }}>
                    Configuration is valid
                  </span>
                </>
              )}
            </div>
            {hasErrors && (
              <div
                className="mt-3 p-3 rounded-lg"
                style={{ backgroundColor: 'var(--danger-light)', border: '1px solid var(--danger)' }}
              >
                <p className="text-sm font-medium" style={{ color: 'var(--danger-text)' }}>
                  Configuration cannot be used until errors are resolved
                </p>
              </div>
            )}
          </div>

          {/* Security Issues */}
          {securityIssues.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-semibold flex items-center" style={{ color: 'var(--text-primary)' }}>
                <svg className="w-5 h-5 mr-2" style={{ color: 'var(--danger)' }} fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                Security Issues
              </h4>
              {securityIssues.map((issue, idx) => (
                <div
                  key={idx}
                  className="p-3 rounded-lg"
                  style={{ ...getLevelStyles(issue.level), borderWidth: '1px', borderStyle: 'solid' }}
                >
                  <div className="flex items-start">
                    <div className="flex-shrink-0">
                      {getLevelIcon(issue.level)}
                    </div>
                    <div className="ml-3 flex-1">
                      <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                        {issue.message}
                      </p>
                      {issue.suggestion && (
                        <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
                          {issue.suggestion}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Validation Messages */}
          {messages.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                Validation
              </h4>
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className="p-3 rounded-lg"
                  style={{ ...getLevelStyles(msg.level), borderWidth: '1px', borderStyle: 'solid' }}
                >
                  <div className="flex items-start">
                    <div className="flex-shrink-0">
                      {getLevelIcon(msg.level)}
                    </div>
                    <div className="ml-3 flex-1">
                      <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                        {msg.message}
                      </p>
                      {msg.suggestion && (
                        <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
                          {msg.suggestion}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Recommendations */}
          {recommendations.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-semibold flex items-center" style={{ color: 'var(--text-primary)' }}>
                <svg className="w-5 h-5 mr-2" style={{ color: 'var(--accent)' }} fill="currentColor" viewBox="0 0 20 20">
                  <path d="M11 3a1 1 0 10-2 0v1a1 1 0 102 0V3zM15.657 5.757a1 1 0 00-1.414-1.414l-.707.707a1 1 0 001.414 1.414l.707-.707zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zM5 10a1 1 0 01-1 1H3a1 1 0 110-2h1a1 1 0 011 1zM8 16v-1h4v1a2 2 0 11-4 0zM12 14c.015-.34.208-.646.477-.859a4 4 0 10-4.954 0c.27.213.462.519.476.859h4.002z" />
                </svg>
                Recommendations
              </h4>
              {recommendations.map((rec, idx) => (
                <div
                  key={idx}
                  className="p-3 rounded-lg"
                  style={{ ...getLevelStyles(rec.level), borderWidth: '1px', borderStyle: 'solid' }}
                >
                  <div className="flex items-start">
                    <div className="flex-shrink-0">
                      {getLevelIcon(rec.level)}
                    </div>
                    <div className="ml-3 flex-1">
                      <p className="text-sm" style={{ color: 'var(--text-primary)' }}>
                        {rec.message}
                      </p>
                      {rec.suggestion && (
                        <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
                          {rec.suggestion}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
