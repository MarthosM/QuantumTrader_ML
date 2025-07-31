# Script PowerShell para iniciar Valkey

Write-Host "========================================"
Write-Host "  Iniciando Valkey para ML Trading"
Write-Host "========================================"
Write-Host ""

# Verificar se Docker está instalado
try {
    $dockerVersion = docker --version
    Write-Host "[OK] Docker encontrado: $dockerVersion"
} catch {
    Write-Host "[ERRO] Docker não encontrado no PATH" -ForegroundColor Red
    Write-Host "Por favor verifique se o Docker Desktop está instalado e rodando"
    Read-Host "Pressione Enter para sair"
    exit 1
}

Write-Host ""
Write-Host "[INFO] Iniciando Valkey..." -ForegroundColor Yellow

# Iniciar Valkey
try {
    docker compose -f docker-compose.valkey.yml up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "[OK] Valkey iniciado com sucesso!" -ForegroundColor Green
        
        Write-Host ""
        Write-Host "Aguardando 5 segundos para inicialização..."
        Start-Sleep -Seconds 5
        
        Write-Host ""
        Write-Host "Testando conexão..."
        docker exec ml-trading-valkey valkey-cli ping
        
        Write-Host ""
        Write-Host "========================================"
        Write-Host "  Comandos úteis:" -ForegroundColor Cyan
        Write-Host "========================================"
        Write-Host "Ver logs:    docker logs ml-trading-valkey"
        Write-Host "Parar:       docker compose -f docker-compose.valkey.yml down"
        Write-Host "Console:     docker exec -it ml-trading-valkey valkey-cli"
        Write-Host "========================================"
    } else {
        Write-Host ""
        Write-Host "[ERRO] Falha ao iniciar Valkey" -ForegroundColor Red
        Write-Host "Verifique se o Docker Desktop está rodando"
    }
} catch {
    Write-Host "[ERRO] $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Read-Host "Pressione Enter para sair"