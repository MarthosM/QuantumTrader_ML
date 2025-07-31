@echo off
echo ========================================
echo   Iniciando Valkey para ML Trading
echo ========================================
echo.

REM Verificar se Docker está rodando
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Docker nao encontrado no PATH
    echo Por favor verifique se o Docker Desktop esta instalado e rodando
    pause
    exit /b 1
)

echo [INFO] Docker encontrado, iniciando Valkey...
echo.

REM Iniciar Valkey
docker compose -f docker-compose.valkey.yml up -d

if errorlevel 1 (
    echo.
    echo [ERRO] Falha ao iniciar Valkey
    echo Verifique se o Docker Desktop esta rodando
) else (
    echo.
    echo [OK] Valkey iniciado com sucesso!
    echo.
    echo Aguardando 5 segundos para inicializacao...
    timeout /t 5 /nobreak >nul
    
    echo.
    echo Testando conexao...
    docker exec ml-trading-valkey valkey-cli ping
    
    echo.
    echo ========================================
    echo   Comandos uteis:
    echo ========================================
    echo Ver logs:    docker logs ml-trading-valkey
    echo Parar:       docker compose -f docker-compose.valkey.yml down
    echo Console:     docker exec -it ml-trading-valkey valkey-cli
    echo ========================================
)

pause