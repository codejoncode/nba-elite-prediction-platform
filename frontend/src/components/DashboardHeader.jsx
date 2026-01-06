import React, { useState } from 'react';

export function DashboardHeader({ onRefresh }) {
  const [isChecking, setIsChecking] = useState(false);
  const [lastCheckTime, setLastCheckTime] = useState(null);
  const [checkStatus, setCheckStatus] = useState('');

  const handleCheckUpdates = async () => {
    if (isChecking) return; // Prevent multiple clicks

    setIsChecking(true);
    setCheckStatus('Checking for updates...');

    try {
      console.log('[DASHBOARD] Checking updates from Flask...');
      
      // Call Flask auto-sync endpoint
      const response = await fetch('http://localhost:5001/api/check-updates', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('[DASHBOARD] Check updates failed:', response.status, errorText);
        setCheckStatus(`Error: ${response.statusText}`);
        setIsChecking(false);
        return;
      }

      const data = await response.json();
      console.log('[DASHBOARD] Check updates successful!', data);

      if (data.success) {
        setCheckStatus('Updates complete!');
        setLastCheckTime(new Date().toLocaleTimeString());
        
        // Trigger parent refresh to reload dashboard data
        if (onRefresh) {
          setTimeout(() => {
            onRefresh();
          }, 500); // Small delay for UI update
        }

        // Clear status message after 3 seconds
        setTimeout(() => {
          setCheckStatus('');
        }, 3000);
      } else {
        setCheckStatus(`Failed: ${data.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('[DASHBOARD] Check updates error:', error);
      setCheckStatus(`Error: ${error.message}`);
    } finally {
      setIsChecking(false);
    }
  };

  return (
    <div style={styles.header}>
      <div style={styles.titleSection}>
        <h1 style={styles.title}>NBA Elite Prediction Dashboard</h1>
        <p style={styles.subtitle}>ML-Powered Game Predictions with Auto-Sync</p>
      </div>

      <div style={styles.controlSection}>
        <button
          onClick={handleCheckUpdates}
          disabled={isChecking}
          style={{
            ...styles.checkUpdatesBtn,
            ...(isChecking ? styles.checkUpdatesBtnLoading : {})
          }}
        >
          {isChecking ? '⟳ Checking Updates...' : '⟳ Check Updates'}
        </button>

        {checkStatus && (
          <div style={{
            ...styles.statusMessage,
            color: checkStatus.includes('complete') ? '#4CAF50' : '#FF9800'
          }}>
            {checkStatus}
          </div>
        )}

        {lastCheckTime && (
          <p style={styles.lastCheckTime}>
            Last checked: {lastCheckTime}
          </p>
        )}
      </div>
    </div>
  );
}

/**
 * Styling for the header and button
 * Can be customized to match your dashboard theme
 */
const styles = {
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '20px',
    backgroundColor: '#f5f5f5',
    borderBottom: '1px solid #ddd',
    marginBottom: '20px',
    flexWrap: 'wrap',
    gap: '20px'
  },
  titleSection: {
    flex: 1,
    minWidth: '300px'
  },
  title: {
    margin: '0 0 5px 0',
    fontSize: '24px',
    fontWeight: 'bold',
    color: '#333'
  },
  subtitle: {
    margin: 0,
    fontSize: '14px',
    color: '#666'
  },
  controlSection: {
    display: 'flex',
    alignItems: 'center',
    gap: '15px',
    flexWrap: 'wrap'
  },
  checkUpdatesBtn: {
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
  checkUpdatesBtnLoading: {
    backgroundColor: '#1976D2',
    opacity: 0.8,
    cursor: 'not-allowed'
  },
  statusMessage: {
    padding: '8px 12px',
    borderRadius: '4px',
    backgroundColor: '#FFF3CD',
    fontSize: '13px',
    fontWeight: '500'
  },
  lastCheckTime: {
    margin: 0,
    fontSize: '12px',
    color: '#999'
  }
};