@echo off
echo ============================================================
echo SETUP VALKEY PARA HMARL - WINDOWS
echo ============================================================
echo.

REM Verificar se Docker está instalado
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] Docker nao esta instalado!
    echo Por favor instale Docker Desktop: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

echo 1. Docker instalado, verificando se esta rodando...

REM Verificar se Docker está rodando
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo    Docker nao esta rodando!
    echo.
    echo 2. Tentando iniciar Docker Desktop...
    
    REM Tentar iniciar Docker Desktop
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    
    echo    Aguardando Docker inicializar (pode levar 30-60 segundos)...
    echo    Por favor aguarde...
    
    REM Aguardar até 60 segundos
    set /a count=0
    :wait_docker
    timeout /t 5 /nobreak >nul
    set /a count+=5
    
    docker info >nul 2>&1
    if %errorlevel% equ 0 (
        echo    [OK] Docker iniciado com sucesso!
        goto docker_ready
    )
    
    if %count% lss 60 (
        echo    Aguardando... %count%/60 segundos
        goto wait_docker
    )
    
    echo    [ERRO] Timeout aguardando Docker iniciar
    echo    Por favor inicie o Docker Desktop manualmente
    pause
    exit /b 1
)

:docker_ready
echo    [OK] Docker esta rodando!
echo.

REM Verificar container existente
echo 3. Verificando containers existentes...
docker ps -a --filter name=valkey-trading --format "table {{.Names}}" | findstr valkey-trading >nul 2>&1
if %errorlevel% equ 0 (
    echo    Container 'valkey-trading' encontrado
    echo    Parando e removendo container antigo...
    docker stop valkey-trading >nul 2>&1
    docker rm valkey-trading >nul 2>&1
    echo    [OK] Container antigo removido
) else (
    echo    [OK] Nenhum container existente
)

REM Criar volume
echo.
echo 4. Criando volume para dados...
docker volume create valkey-data >nul 2>&1
echo    [OK] Volume 'valkey-data' criado/verificado

REM Iniciar Valkey
echo.
echo 5. Baixando e iniciando Valkey...
docker run -d ^
    --name valkey-trading ^
    -p 6379:6379 ^
    -v valkey-data:/data ^
    --restart unless-stopped ^
    valkey/valkey:latest ^
    --maxmemory 2gb ^
    --maxmemory-policy allkeys-lru ^
    --save 60 1000 ^
    --save 300 100 ^
    --save 900 1

if %errorlevel% neq 0 (
    echo    [ERRO] Falha ao iniciar Valkey!
    docker logs valkey-trading
    pause
    exit /b 1
)

echo    [OK] Container Valkey iniciado!

REM Aguardar Valkey ficar pronto
echo.
echo 6. Aguardando Valkey inicializar...
timeout /t 3 /nobreak >nul

docker exec valkey-trading redis-cli ping >nul 2>&1
if %errorlevel% neq 0 (
    echo    Aguardando mais um pouco...
    timeout /t 5 /nobreak >nul
)

REM Testar conexão
echo.
echo 7. Testando conexao...
docker exec valkey-trading redis-cli ping
if %errorlevel% equ 0 (
    echo    [OK] Valkey respondendo!
) else (
    echo    [ERRO] Valkey nao esta respondendo
    echo    Verifique os logs: docker logs valkey-trading
    pause
    exit /b 1
)

REM Mostrar informações
echo.
echo ============================================================
echo VALKEY INSTALADO COM SUCESSO!
echo ============================================================
echo.
echo Informacoes do container:
docker ps --filter name=valkey-trading --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.
echo Comandos uteis:
echo   - Status: docker ps --filter name=valkey-trading
echo   - Logs: docker logs valkey-trading
echo   - CLI: docker exec -it valkey-trading redis-cli
echo   - Parar: docker stop valkey-trading
echo   - Iniciar: docker start valkey-trading
echo.
echo Proximo passo:
echo   Execute: python -m pytest tests/test_zmq_valkey_infrastructure.py -v
echo.
pause