// API base URL
const API_BASE = '/api';

// State
let statusInterval = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadConfig();
    updateStatus();
    
    // Set up event listeners
    document.getElementById('startBtn').addEventListener('click', startSystem);
    document.getElementById('stopBtn').addEventListener('click', stopSystem);
    document.getElementById('refreshBtn').addEventListener('click', updateStatus);
    document.getElementById('saveConfigBtn').addEventListener('click', saveConfig);
    document.getElementById('runBacktestBtn').addEventListener('click', runBacktest);
    document.getElementById('brokerageProvider').addEventListener('change', handleBrokerageChange);
    document.getElementById('tradingMode').addEventListener('change', handleTradingModeChange);
    document.getElementById('testConnectionBtn').addEventListener('click', testConnection);
    document.getElementById('saveCredentialsBtn').addEventListener('click', saveCredentials);
    
    // Set default dates for backtest
    const today = new Date();
    const oneYearAgo = new Date(today.getFullYear() - 1, today.getMonth(), today.getDate());
    document.getElementById('backtestStartDate').valueAsDate = oneYearAgo;
    document.getElementById('backtestEndDate').valueAsDate = today;
});

// Load configuration
async function loadConfig() {
    try {
        const response = await fetch(`${API_BASE}/config`);
        const data = await response.json();
        
        if (data.success) {
            const config = data.config;
            document.getElementById('tradingMode').value = config.trading.mode;
            document.getElementById('symbols').value = config.trading.symbols.join(', ');
            document.getElementById('updateInterval').value = config.trading.update_interval_seconds;
            document.getElementById('initialCapital').value = config.trading.initial_capital;
        }
    } catch (error) {
        console.error('Error loading config:', error);
    }
}

// Save configuration
async function saveConfig() {
    const btn = document.getElementById('saveConfigBtn');
    btn.disabled = true;
    btn.textContent = 'Saving...';
    
    try {
        const symbols = document.getElementById('symbols').value
            .split(',')
            .map(s => s.trim())
            .filter(s => s);
        
        const config = {
            trading: {
                mode: document.getElementById('tradingMode').value,
                symbols: symbols,
                update_interval_seconds: parseInt(document.getElementById('updateInterval').value),
                initial_capital: parseFloat(document.getElementById('initialCapital').value)
            },
            risk: {
                max_position_size_pct: 10.0,
                max_daily_loss_pct: 5.0,
                max_portfolio_leverage: 1.0
            },
            strategies: [
                {
                    name: 'MovingAverageCrossover',
                    params: {
                        fast_period: 10,
                        slow_period: 30,
                        symbols: symbols
                    }
                }
            ],
            data: {
                provider: 'yfinance',
                cache_enabled: true,
                cache_dir: './data/cache'
            },
            brokerage: {
                provider: 'simulated'
            }
        };
        
        const response = await fetch(`${API_BASE}/config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert('Configuration saved successfully!', 'success');
        } else {
            showAlert(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showAlert(`Error: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Save Configuration';
    }
}

// Start system
async function startSystem() {
    const btn = document.getElementById('startBtn');
    btn.disabled = true;
    btn.textContent = 'Starting...';
    
    try {
        const response = await fetch(`${API_BASE}/start`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showAlert('Trading system started!', 'success');
            document.getElementById('stopBtn').disabled = false;
            
            // Start polling for status
            if (statusInterval) clearInterval(statusInterval);
            statusInterval = setInterval(updateStatus, 5000);
            updateStatus();
        } else {
            showAlert(`Error: ${data.error}`, 'error');
            btn.disabled = false;
            btn.textContent = 'Start System';
        }
    } catch (error) {
        showAlert(`Error: ${error.message}`, 'error');
        btn.disabled = false;
        btn.textContent = 'Start System';
    }
}

// Stop system
async function stopSystem() {
    const btn = document.getElementById('stopBtn');
    btn.disabled = true;
    btn.textContent = 'Stopping...';
    
    try {
        const response = await fetch(`${API_BASE}/stop`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showAlert('Trading system stopped!', 'success');
            document.getElementById('startBtn').disabled = false;
            document.getElementById('startBtn').textContent = 'Start System';
            
            // Stop polling
            if (statusInterval) {
                clearInterval(statusInterval);
                statusInterval = null;
            }
            
            updateStatus();
        } else {
            showAlert(`Error: ${data.error}`, 'error');
            btn.disabled = false;
            btn.textContent = 'Stop System';
        }
    } catch (error) {
        showAlert(`Error: ${error.message}`, 'error');
        btn.disabled = false;
        btn.textContent = 'Stop System';
    }
}

// Update status
async function updateStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const status = await response.json();
        
        // Update status badge
        const statusBadge = document.getElementById('systemStatus');
        const statusText = statusBadge.querySelector('.status-text');
        
        if (status.running) {
            statusBadge.classList.add('running');
            statusText.textContent = `Running (${status.mode})`;
        } else {
            statusBadge.classList.remove('running');
            statusText.textContent = 'Stopped';
        }
        
        // Update portfolio stats
        if (status.portfolio) {
            const p = status.portfolio;
            document.getElementById('totalValue').textContent = formatCurrency(p.total_value);
            document.getElementById('cash').textContent = formatCurrency(p.cash);
            document.getElementById('positionsValue').textContent = formatCurrency(p.positions_value);
            
            const pnlElement = document.getElementById('unrealizedPnl');
            pnlElement.textContent = formatCurrency(p.unrealized_pnl);
            pnlElement.className = 'stat-value ' + (p.unrealized_pnl >= 0 ? 'positive' : 'negative');
            
            const returnElement = document.getElementById('totalReturn');
            returnElement.textContent = formatPercent(p.total_return);
            returnElement.className = 'stat-value ' + (p.total_return >= 0 ? 'positive' : 'negative');
            
            document.getElementById('numPositions').textContent = p.num_positions;
        }
        
        // Update positions table
        if (status.positions && status.positions.length > 0) {
            const tbody = document.getElementById('positionsBody');
            tbody.innerHTML = status.positions.map(pos => `
                <tr>
                    <td><strong>${pos.symbol}</strong></td>
                    <td>${pos.quantity}</td>
                    <td>${formatCurrency(pos.avg_price)}</td>
                    <td>${formatCurrency(pos.current_price)}</td>
                    <td>${formatCurrency(pos.market_value)}</td>
                    <td class="${pos.unrealized_pnl >= 0 ? 'positive' : 'negative'}">
                        ${formatCurrency(pos.unrealized_pnl)}
                    </td>
                    <td class="${pos.unrealized_pnl_pct >= 0 ? 'positive' : 'negative'}">
                        ${formatPercent(pos.unrealized_pnl_pct)}
                    </td>
                </tr>
            `).join('');
        } else {
            document.getElementById('positionsBody').innerHTML = 
                '<tr><td colspan="7" class="empty-state">No positions</td></tr>';
        }
    } catch (error) {
        console.error('Error updating status:', error);
    }
}

// Run backtest
async function runBacktest() {
    const btn = document.getElementById('runBacktestBtn');
    btn.disabled = true;
    btn.textContent = 'Running...';
    
    const resultsDiv = document.getElementById('backtestResults');
    resultsDiv.innerHTML = '<p>Running backtest...</p>';
    resultsDiv.classList.add('show');
    
    try {
        const symbols = document.getElementById('backtestSymbols').value
            .split(',')
            .map(s => s.trim())
            .filter(s => s);
        
        const params = {
            start_date: document.getElementById('backtestStartDate').value,
            end_date: document.getElementById('backtestEndDate').value,
            symbols: symbols,
            initial_capital: parseFloat(document.getElementById('initialCapital').value)
        };
        
        const response = await fetch(`${API_BASE}/backtest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        
        const data = await response.json();
        
        if (data.success) {
            const result = data.result;
            resultsDiv.innerHTML = `
                <h3>Backtest Results</h3>
                <div class="result-grid">
                    <div class="stat">
                        <div class="stat-label">Total Return</div>
                        <div class="stat-value ${result.total_return >= 0 ? 'positive' : 'negative'}">
                            ${formatPercent(result.total_return)}
                        </div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Sharpe Ratio</div>
                        <div class="stat-value">${result.sharpe_ratio.toFixed(2)}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Max Drawdown</div>
                        <div class="stat-value negative">${formatPercent(result.max_drawdown)}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Total Trades</div>
                        <div class="stat-value">${result.total_trades}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Win Rate</div>
                        <div class="stat-value">${formatPercent(result.win_rate)}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Final Value</div>
                        <div class="stat-value">${formatCurrency(result.final_value)}</div>
                    </div>
                </div>
            `;
        } else {
            resultsDiv.innerHTML = `<div class="alert alert-error">${data.error}</div>`;
        }
    } catch (error) {
        resultsDiv.innerHTML = `<div class="alert alert-error">${error.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Backtest';
    }
}

// Utility functions
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2
    }).format(value);
}

function formatPercent(value) {
    return `${(value * 100).toFixed(2)}%`;
}

function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;
    
    document.querySelector('.container').insertBefore(
        alertDiv,
        document.querySelector('.container').firstChild
    );
    
    setTimeout(() => alertDiv.remove(), 5000);
}

// Handle brokerage provider change
function handleBrokerageChange() {
    const provider = document.getElementById('brokerageProvider').value;
    const credentialsSection = document.getElementById('credentialsSection');
    const testBtn = document.getElementById('testConnectionBtn');
    const saveBtn = document.getElementById('saveCredentialsBtn');
    
    if (provider !== 'simulated') {
        credentialsSection.style.display = 'block';
        testBtn.style.display = 'inline-block';
        saveBtn.style.display = 'inline-block';
    } else {
        credentialsSection.style.display = 'none';
        testBtn.style.display = 'none';
        saveBtn.style.display = 'none';
    }
}

// Handle trading mode change
function handleTradingModeChange() {
    const mode = document.getElementById('tradingMode').value;
    const warning = document.getElementById('liveWarning');
    
    if (mode === 'live') {
        warning.style.display = 'block';
    } else {
        warning.style.display = 'none';
    }
}

// Test brokerage connection
async function testConnection() {
    const btn = document.getElementById('testConnectionBtn');
    const statusDiv = document.getElementById('connectionStatus');
    
    btn.disabled = true;
    btn.textContent = 'Testing...';
    statusDiv.className = 'status-message info show';
    statusDiv.textContent = 'Testing connection to brokerage...';
    
    try {
        const provider = document.getElementById('brokerageProvider').value;
        const apiKey = document.getElementById('apiKey').value;
        const apiSecret = document.getElementById('apiSecret').value;
        
        if (!apiKey || !apiSecret) {
            throw new Error('Please enter both API key and secret');
        }
        
        const response = await fetch(`${API_BASE}/test-connection`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: provider,
                api_key: apiKey,
                api_secret: apiSecret
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            statusDiv.className = 'status-message success show';
            statusDiv.innerHTML = `
                <strong>✓ Connection Successful!</strong><br>
                Account ID: ${data.account_id || 'N/A'}<br>
                Balance: ${data.balance ? formatCurrency(data.balance) : 'N/A'}
            `;
            showAlert('Successfully connected to brokerage!', 'success');
        } else {
            throw new Error(data.error || 'Connection failed');
        }
    } catch (error) {
        statusDiv.className = 'status-message error show';
        statusDiv.innerHTML = `<strong>✗ Connection Failed</strong><br>${error.message}`;
        showAlert(`Connection failed: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Test Connection';
    }
}

// Save brokerage credentials
async function saveCredentials() {
    const btn = document.getElementById('saveCredentialsBtn');
    const statusDiv = document.getElementById('connectionStatus');
    
    btn.disabled = true;
    btn.textContent = 'Saving...';
    
    try {
        const provider = document.getElementById('brokerageProvider').value;
        const apiKey = document.getElementById('apiKey').value;
        const apiSecret = document.getElementById('apiSecret').value;
        const saveToFile = document.getElementById('saveCredentials').checked;
        
        if (!apiKey || !apiSecret) {
            throw new Error('Please enter both API key and secret');
        }
        
        const response = await fetch(`${API_BASE}/save-credentials`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: provider,
                api_key: apiKey,
                api_secret: apiSecret,
                save_to_file: saveToFile
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            statusDiv.className = 'status-message success show';
            statusDiv.innerHTML = '<strong>✓ Credentials Saved!</strong><br>You can now start the system in live mode.';
            showAlert('Credentials saved successfully!', 'success');
            
            // Clear password field for security
            if (!saveToFile) {
                document.getElementById('apiSecret').value = '';
            }
        } else {
            throw new Error(data.error || 'Failed to save credentials');
        }
    } catch (error) {
        statusDiv.className = 'status-message error show';
        statusDiv.innerHTML = `<strong>✗ Save Failed</strong><br>${error.message}`;
        showAlert(`Failed to save: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Save Credentials';
    }
}
