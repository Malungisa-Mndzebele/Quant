"""AI Trading Agent - Configuration"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Application Configuration (for backward compatibility with app.py)
APP_TITLE = "Quantitative Trading System"
APP_ICON = "📈"
APP_LAYOUT = "wide"

# API Configuration (from root config.py)
TDA_API_KEY = os.getenv('TDA_API_KEY', '')
USE_MOCK_DATA = os.getenv('USE_MOCK_DATA', 'True').lower() == 'true'

# Caching Configuration
CACHE_TTL_SECONDS = 300

# Import settings from config modules
from config.settings import *
from config.logging_config import *
