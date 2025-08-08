"""
Configuração do Monitor - QuantumTrader ML
"""

# Tipo de monitor a usar
# 1 = GUI Desktop (Tkinter)
# 2 = Web (Flask + Socket.IO)
MONITOR_TYPE = 1

# Configurações do Monitor Web
WEB_PORT = 5000
WEB_AUTO_OPEN_BROWSER = True

# Configurações do Monitor GUI
GUI_WIDTH = 1200
GUI_HEIGHT = 800
GUI_UPDATE_INTERVAL = 100  # ms

# Configurações de exibição
SHOW_PREDICTIONS_CHART = True
SHOW_PNL_CHART = True
SHOW_TRADES_HISTORY = True
SHOW_LOGS = True
MAX_LOG_LINES = 200
MAX_PREDICTIONS = 100
MAX_TRADES = 50

# Cores do tema
THEME = {
    'background': '#1e1e1e',
    'foreground': '#ffffff',
    'card_bg': '#2d2d2d',
    'positive': '#4CAF50',
    'negative': '#f44336',
    'neutral': '#FFC107',
    'ml_color': '#44aaff',
    'trade_color': '#ffff44'
}