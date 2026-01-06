import React, { useState } from 'react';

/**
 * DashboardControls Component
 * 
 * Single button component that triggers Flask auto-sync.
 * Shows loading state, success/error messages, and last refresh time.
 * 
 * Props:
 * - onRefresh: Function to reload dashboard data after sync
 * - loading: Boolean showing if data is currently loading
 * - lastRefresh: Date object of last refresh time
 */
const DashboardControls = ({ onRefresh, loading, lastRefresh }) => {
  const [isCheckingUpdates, setIsCheckingUpdates] = useState(false);
  const [checkStatus, setCheckStatus] = useState('');
  const [checkError, setCheckError] = useState('');

  const handleCheckUpdates = async () => {
    if (isCheckingUpdates) return; // Prevent double-clicks

    setIsCheckingUpdates(true);
    setCheckStatus('Checking for updates...');
    setCheckError('');

    try {
      console.log('[DashboardControls] Triggering Flask sync...');

      // Call Flask auto-sync endpoint
      const response = await fetch('http://localhost:5001/api/check-updates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`Sync failed: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('[DashboardControls] Sync successful:', data);

      if (data.success) {
        setCheckStatus('✓ Updates complete!');
        
        // Reload dashboard data
        if (onRefresh) {
          setTimeout(() => {
            onRefresh();
          }, 500);
        }

        // Clear status after 3 seconds
        setTimeout(() => {
          setCheckStatus('');
        }, 3000);
      } else {
        throw new Error(data.error || 'Unknown sync error');
      }
    } catch (error) {
      console.error('[DashboardControls] Sync error:', error);
      setCheckError(`Error: ${error.message}`);
      setCheckStatus('');
    } finally {
      setIsCheckingUpdates(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.left}>
        <h2 style={styles.title}>Dashboard Controls</h2>
      </div>

      <div style={styles.right}>
        {/* Check Updates Button */}
        <button
          onClick={handleCheckUpdates}
          disabled={isCheckingUpdates || loading}
          style={{
            ...styles.button,
            ...(isCheckingUpdates || loading ? styles.buttonDisabled : {})
          }}
        >
          {isCheckingUpdates ? '⟳ Syncing...' : '⟳ Check Updates'}
        </button>

        {/* Status Messages */}
        {checkStatus && (
          <span style={{ ...styles.status, color: '#4CAF50' }}>
            {checkStatus}
          </span>
        )}

        {checkError && (
          <span style={{ ...styles.status, color: '#FF6B6B' }}>
            {checkError}
          </span>
        )}

        {/* Last Refresh Time */}
        {lastRefresh && (
          <span style={styles.lastRefresh}>
            Last updated: {lastRefresh.toLocaleTimeString()}
          </span>
        )}
      </div>
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '20px',
    backgroundColor: 'rgba(15, 23, 42, 0.5)',
    borderRadius: '8px',
    border: '1px solid rgba(148, 163, 184, 0.2)',
    marginBottom: '30px',
    flexWrap: 'wrap',
    gap: '20px'
  },
  left: {
    flex: 1,
    minWidth: '200px'
  },
  title: {
    margin: 0,
    fontSize: '18px',
    fontWeight: '600',
    color: '#e2e8f0'
  },
  right: {
    display: 'flex',
    alignItems: 'center',
    gap: '15px',
    flexWrap: 'wrap'
  },
  button: {
    padding: '10px 20px',
    backgroundColor: '#2196F3',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500',
    transition: 'all 0.3s ease',
    whiteSpace: 'nowrap'
  },
  buttonDisabled: {
    backgroundColor: '#1976D2',
    opacity: 0.6,
    cursor: 'not-allowed'
  },
  status: {
    fontSize: '13px',
    fontWeight: '500',
    padding: '6px 12px',
    borderRadius: '4px',
    backgroundColor: 'rgba(255, 255, 255, 0.1)'
  },
  lastRefresh: {
    fontSize: '12px',
    color: '#94a3b8'
  }
};

export default DashboardControls;