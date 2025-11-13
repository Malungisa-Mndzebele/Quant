"""Web interface for the Quantitative Trading System."""
import os
import json
import threading
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

from src.trading_system import TradingSystem
from src.utils.config_loader import ConfigLoader

app = Flask(__name__)
CORS(app)

# Global state
trading_system = None
system_thread = None
system_running = False
system_status = {
    'running': False,
    'mode': None,
    'portfolio': {},
    'positions': [],
    'recent_trades': [],
    'performance': {},
    'last_update': None
}

logger = logging.getLogger(__name__)


@app.route('/')
def index():
    """Render main dashboard."""
    return render_template('index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    try:
        config = ConfigLoader.load_config('config.yaml')
        return jsonify({
            'success': True,
            'config': config
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration."""
    try:
        new_config = request.json
        
        # Save to config.yaml
        import yaml
        with open('config.yaml', 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)
        
        return jsonify({
            'success': True,
            'message': 'Configuration updated successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/start', methods=['POST'])
def start_system():
    """Start the trading system."""
    global trading_system, system_thread, system_running
    
    if system_running:
        return jsonify({
            'success': False,
            'error': 'System is already running'
        }), 400
    
    try:
        # Load configuration
        config = ConfigLoader.load_config('config.yaml')
        
        # Initialize trading system
        trading_system = TradingSystem(config)
        
        # Start in a separate thread
        system_running = True
        system_thread = threading.Thread(target=run_trading_loop, daemon=True)
        system_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Trading system started successfully'
        })
    except Exception as e:
        system_running = False
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stop', methods=['POST'])
def stop_system():
    """Stop the trading system."""
    global system_running, trading_system
    
    if not system_running:
        return jsonify({
            'success': False,
            'error': 'System is not running'
        }), 400
    
    try:
        system_running = False
        if trading_system:
            trading_system.stop()
        
        return jsonify({
            'success': True,
            'message': 'Trading system stopped successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current system status."""
    global system_status, trading_system
    
    if trading_system and system_running:
        try:
            # Update status from trading system
            portfolio = trading_system.portfolio
            account_info = trading_system.brokerage.get_account_info()
            positions = trading_system.brokerage.get_positions()
            
            system_status.update({
                'running': True,
                'mode': trading_system.config.trading.mode.value,
                'portfolio': {
                    'total_value': portfolio.get_total_value(),
                    'cash': portfolio.cash,
                    'positions_value': portfolio.get_positions_value(),
                    'unrealized_pnl': portfolio.get_unrealized_pnl(),
                    'total_return': portfolio.get_total_return(),
                    'num_positions': len(portfolio.positions)
                },
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'quantity': pos.quantity,
                        'avg_price': pos.average_price,
                        'current_price': pos.current_price,
                        'market_value': pos.market_value,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'unrealized_pnl_pct': pos.unrealized_pnl_pct
                    }
                    for pos in positions
                ],
                'account': {
                    'account_id': account_info.account_id,
                    'cash': account_info.cash_balance,
                    'buying_power': account_info.buying_power,
                    'portfolio_value': account_info.portfolio_value
                },
                'last_update': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    return jsonify(system_status)


@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run a backtest with given parameters."""
    try:
        params = request.json
        
        from src.backtesting.backtester import Backtester
        from src.data.yfinance_provider import YFinanceProvider
        from datetime import date
        
        # Parse dates
        start_date = date.fromisoformat(params['start_date'])
        end_date = date.fromisoformat(params['end_date'])
        symbols = params['symbols']
        
        # Load config and create backtester
        config = ConfigLoader.load_config('config.yaml')
        data_provider = YFinanceProvider()
        
        backtester = Backtester(
            config=config,
            data_provider=data_provider,
            initial_capital=params.get('initial_capital', 100000)
        )
        
        # Run backtest
        result = backtester.run(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        return jsonify({
            'success': True,
            'result': result.to_dict()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def run_trading_loop():
    """Run the trading system loop."""
    global trading_system, system_running
    
    try:
        config = trading_system.config
        symbols = config.trading.symbols
        interval = config.trading.update_interval_seconds
        
        trading_system.run(symbols=symbols, interval_seconds=interval)
    except Exception as e:
        logger.error(f"Trading loop error: {e}")
        system_running = False


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
