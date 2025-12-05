# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2025-12-05

### Added
- **Streamlit Web Application** (`app.py`)
  - Interactive web-based UI for option pricing and analysis
  - Four main pages: Option Pricing, Market Data, Implied Volatility, About
  - Real-time calculation with visual feedback
  - Responsive design with custom CSS styling
  - Interactive charts using Plotly
  - Download capabilities for market data

- **Mock Data Support**
  - Complete mock data generation for option chains and historical prices
  - Enables full functionality without TDAmeritrade API key
  - Configurable via `USE_MOCK_DATA` environment variable
  - Realistic data generation with proper option Greeks

- **Configuration System** (`config.py`)
  - Centralized configuration management
  - Model parameter definitions and constraints
  - Display settings and formatting options
  - Environment variable integration via python-dotenv

- **Service Layer**
  - `services/api_service.py` - Enhanced API wrapper with error handling
  - `services/pricing_service.py` - Unified pricing interface for all models
  - Mock data generators for testing and demos
  - Comprehensive error handling with custom exceptions

- **Utility Modules**
  - `utils/formatters.py` - Consistent formatting for prices, Greeks, and percentages
  - `utils/validators.py` - Input validation with detailed error messages
  - Type-safe interfaces with proper error handling

- **Comprehensive Test Suite**
  - `tests/test_api_service.py` - API error handling tests (18 tests)
  - `tests/test_pricing_service.py` - Pricing calculation tests (4 tests)
  - `tests/test_formatters.py` - Formatting utility tests (13 tests)
  - `tests/test_validators.py` - Validation logic tests (11 tests)
  - Total: 52 passing tests with pytest configuration

- **Demo Scripts**
  - `demo.py` - Basic option pricing demonstration
  - `demo_mock_api.py` - Mock data API demonstration

- **Development Tools**
  - `pytest.ini` - Pytest configuration with proper Python path
  - `requirements.txt` - Complete dependency management
  - Updated `.gitignore` for virtual environments and IDE files

### Changed
- Enhanced README with comprehensive documentation
- Improved error messages and logging throughout
- Better type hints and docstrings
- Organized project structure with clear separation of concerns

### Fixed
- Module import issues with pytest by adding `pytest.ini`
- API connection errors with mock data fallback
- Test isolation with proper fixture management
- Line ending consistency in git repository

### Technical Details
- **Dependencies Added**: streamlit, plotly, python-dotenv
- **Testing Framework**: pytest with asyncio and hypothesis support
- **Code Quality**: All 52 tests passing
- **Architecture**: Clean separation of services, utilities, and UI layers

## [0.5.1] - 2023-XX-XX (Original)

### Added
- Core option pricing models (Black-Scholes, Merton, Black-76, etc.)
- American option pricing using Bjerksund-Stensland approximation
- Complete Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Implied volatility calculators for European and American options
- TDAmeritrade API integration
- Basic documentation and examples

---

## Migration Notes

### Upgrading from 0.5.1 to 0.6.0

**New Requirements:**
```bash
pip install streamlit plotly python-dotenv
```

**Configuration Setup:**
Create a `.env` file in the project root:
```ini
TDA_API_KEY=your_key_here  # Optional, can use mock data
USE_MOCK_DATA=True  # Set to False to use real API
```

**Running the Application:**
```bash
streamlit run app.py
```

**Running Tests:**
```bash
pytest
```

### Breaking Changes
None. All existing functionality remains backward compatible.

### New Features Usage

**Mock Data Mode (Default):**
```python
from services.api_service import fetch_option_chain
chain = fetch_option_chain("AAPL")  # Works without API key
```

**Pricing Service:**
```python
from services.pricing_service import calculate_option_price

params = {
    'option_type': 'c',
    'fs': 100,
    'x': 100,
    't': 1.0,
    'r': 0.05,
    'v': 0.20
}
result = calculate_option_price('black_scholes', params)
print(f"Price: {result.value}, Delta: {result.delta}")
```

**Web Interface:**
Simply run `streamlit run app.py` and access at http://localhost:8501

---

## Contributors

- **Original Author**: Davis Edwards
- **Enhancements & UI**: Daniel Rojas, Malungisa Mndzebele
- **Base Library**: [optlib](https://github.com/dbrojas/optlib)

## License

MIT License - See LICENSE file for details
