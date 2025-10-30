import React from 'react';
import { useContainerStore } from '../../store/useContainerStore';
import type { ValidationLevel } from '../../types';

export const ValidationDisplay: React.FC = () => {
  const { validationResult } = useContainerStore();
  
  if (!validationResult) {
    return null;
  }
  
  const { messages, securityIssues, recommendations } = validationResult;
  
  const getLevelIcon = (level: ValidationLevel) => {
    switch (level) {
      case 'error':
        return (
          <svg className="w-5 h-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
        );
      case 'warning':
        return (
          <svg className="w-5 h-5 text-yellow-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
        );
      case 'info':
        return (
          <svg className="w-5 h-5 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
        );
      case 'success':
        return (
          <svg className="w-5 h-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        );
    }
  };
  
  const getLevelColor = (level: ValidationLevel) => {
    switch (level) {
      case 'error':
        return 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800';
      case 'warning':
        return 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800';
      case 'info':
        return 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800';
      case 'success':
        return 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800';
    }
  };
  
  const hasErrors = messages.some(m => m.level === 'error') || securityIssues.some(m => m.level === 'error');
  const hasWarnings = messages.some(m => m.level === 'warning') || securityIssues.some(m => m.level === 'warning');
  
  return (
    <div className="space-y-3">
      {/* Overall Status */}
      <div className={`p-4 rounded-lg border ${
        hasErrors
          ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
          : hasWarnings
          ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
          : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
      }`}>
        <div className="flex items-center">
          {hasErrors ? (
            <>
              <svg className="w-6 h-6 text-red-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <span className="font-medium text-red-900 dark:text-red-200">
                Configuration has errors
              </span>
            </>
          ) : hasWarnings ? (
            <>
              <svg className="w-6 h-6 text-yellow-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <span className="font-medium text-yellow-900 dark:text-yellow-200">
                Configuration valid with warnings
              </span>
            </>
          ) : (
            <>
              <svg className="w-6 h-6 text-green-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <span className="font-medium text-green-900 dark:text-green-200">
                Configuration is valid
              </span>
            </>
          )}
        </div>
      </div>
      
      {/* Security Issues */}
      {securityIssues.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 flex items-center">
            <svg className="w-5 h-5 text-red-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            Security Issues
          </h4>
          {securityIssues.map((issue, idx) => (
            <div key={idx} className={`p-3 rounded-md border ${getLevelColor(issue.level)}`}>
              <div className="flex items-start">
                <div className="flex-shrink-0">
                  {getLevelIcon(issue.level)}
                </div>
                <div className="ml-3 flex-1">
                  <p className="text-sm text-gray-800 dark:text-gray-200 font-medium">
                    {issue.message}
                  </p>
                  {issue.suggestion && (
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      💡 {issue.suggestion}
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
          <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Validation
          </h4>
          {messages.map((msg, idx) => (
            <div key={idx} className={`p-3 rounded-md border ${getLevelColor(msg.level)}`}>
              <div className="flex items-start">
                <div className="flex-shrink-0">
                  {getLevelIcon(msg.level)}
                </div>
                <div className="ml-3 flex-1">
                  <p className="text-sm text-gray-800 dark:text-gray-200 font-medium">
                    {msg.message}
                  </p>
                  {msg.suggestion && (
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      💡 {msg.suggestion}
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
          <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 flex items-center">
            <svg className="w-5 h-5 text-blue-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path d="M11 3a1 1 0 10-2 0v1a1 1 0 102 0V3zM15.657 5.757a1 1 0 00-1.414-1.414l-.707.707a1 1 0 001.414 1.414l.707-.707zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zM5 10a1 1 0 01-1 1H3a1 1 0 110-2h1a1 1 0 011 1zM8 16v-1h4v1a2 2 0 11-4 0zM12 14c.015-.34.208-.646.477-.859a4 4 0 10-4.954 0c.27.213.462.519.476.859h4.002z" />
            </svg>
            Recommendations
          </h4>
          {recommendations.map((rec, idx) => (
            <div key={idx} className={`p-3 rounded-md border ${getLevelColor(rec.level)}`}>
              <div className="flex items-start">
                <div className="flex-shrink-0">
                  {getLevelIcon(rec.level)}
                </div>
                <div className="ml-3 flex-1">
                  <p className="text-sm text-gray-800 dark:text-gray-200">
                    {rec.message}
                  </p>
                  {rec.suggestion && (
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      💡 {rec.suggestion}
                    </p>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
