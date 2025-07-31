@echo off
REM Setup ZMQ + Valkey para Windows
REM Executa o setup e configura o ambiente

echo ========================================
echo   Setup ZMQ + Valkey - ML Trading
echo ========================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado no PATH
    echo Por favor instale Python 3.8+ primeiro
    pause
    exit /b 1
)

REM Executar setup
echo [INFO] Executando setup...
python setup_zmq_valkey.py

if errorlevel 1 (
    echo.
    echo [AVISO] Setup concluido com avisos
    echo Verifique os logs acima
) else (
    echo.
    echo [OK] Setup concluido com sucesso!
)

echo.
echo ========================================
echo   Comandos Uteis:
echo ========================================
echo.
echo 1. Iniciar Valkey:
echo    docker-compose -f docker-compose.valkey.yml up -d
echo.
echo 2. Testar ZMQ:
echo    python scripts\test_zmq_publisher.py
echo.
echo 3. Testar Valkey:
echo    python scripts\test_valkey_time_travel.py
echo.
echo 4. Monitor:
echo    python scripts\monitor_zmq_valkey.py
echo.
echo 5. Parar Valkey:
echo    docker-compose -f docker-compose.valkey.yml down
echo.
echo ========================================

pause