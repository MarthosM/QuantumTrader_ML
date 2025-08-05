# Script PowerShell para iniciar coleta contínua de Book
# Garante que o ambiente virtual está ativado

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  INICIANDO COLETA CONTÍNUA DE BOOK" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar se está no diretório correto
$expectedPath = "C:\Users\marth\OneDrive\Programacao\Python\Projetos\QuantumTrader_ML"
if ($PWD.Path -ne $expectedPath) {
    Write-Host "Mudando para diretório do projeto..." -ForegroundColor Yellow
    Set-Location $expectedPath
}

# Ativar ambiente virtual
Write-Host "Ativando ambiente virtual..." -ForegroundColor Green
& .\.venv\Scripts\Activate.ps1

# Verificar se ativou corretamente
if ($env:VIRTUAL_ENV) {
    Write-Host "✓ Ambiente virtual ativado" -ForegroundColor Green
} else {
    Write-Host "✗ Erro ao ativar ambiente virtual" -ForegroundColor Red
    exit 1
}

# Mostrar informações do sistema
Write-Host ""
Write-Host "Informações do Sistema:" -ForegroundColor Cyan
Write-Host "  Data/Hora: $(Get-Date -Format 'dd/MM/yyyy HH:mm:ss')"
Write-Host "  Python: $(python --version)"
Write-Host "  Diretório: $PWD"
Write-Host ""

# Verificar se há coleta em andamento
$pidFile = "collection_manager.pid"
if (Test-Path $pidFile) {
    $pid = Get-Content $pidFile
    $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host "⚠️  Coleta já está em execução (PID: $pid)" -ForegroundColor Yellow
        $response = Read-Host "Deseja parar a coleta atual? (S/N)"
        if ($response -eq "S" -or $response -eq "s") {
            Stop-Process -Id $pid -Force
            Remove-Item $pidFile
            Write-Host "✓ Coleta anterior encerrada" -ForegroundColor Green
        } else {
            Write-Host "Mantendo coleta atual" -ForegroundColor Yellow
            exit 0
        }
    } else {
        Remove-Item $pidFile
    }
}

# Opções de execução
Write-Host "Escolha o modo de execução:" -ForegroundColor Cyan
Write-Host "  1. Coleta Contínua com Gerenciador (recomendado)"
Write-Host "  2. Coleta Contínua Simples"
Write-Host "  3. Coleta por Tempo Definido"
Write-Host ""
$choice = Read-Host "Opção (1-3)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Iniciando Coleta Contínua com Gerenciador..." -ForegroundColor Green
        Write-Host "O gerenciador irá:" -ForegroundColor Yellow
        Write-Host "  - Reiniciar automaticamente em caso de falha"
        Write-Host "  - Monitorar horário de mercado"
        Write-Host "  - Salvar logs em continuous_collection.log"
        Write-Host ""
        Write-Host "Pressione Ctrl+C para parar" -ForegroundColor Yellow
        Write-Host ""
        
        python start_continuous_collection.py
    }
    "2" {
        Write-Host ""
        Write-Host "Iniciando Coleta Contínua Simples..." -ForegroundColor Green
        Write-Host "A coleta irá rodar até o fim do pregão ou Ctrl+C" -ForegroundColor Yellow
        Write-Host ""
        
        python book_collector_continuous.py
    }
    "3" {
        Write-Host ""
        $minutes = Read-Host "Por quantos minutos deseja coletar? (padrão: 120)"
        if (-not $minutes) { $minutes = "120" }
        
        Write-Host "Iniciando Coleta por $minutes minutos..." -ForegroundColor Green
        Write-Host ""
        
        # Modificar temporariamente o book_collector.py para rodar por tempo definido
        python book_collector.py
    }
    default {
        Write-Host "Opção inválida!" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "         COLETA FINALIZADA" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Dados salvos em: data\realtime\book\$(Get-Date -Format 'yyyyMMdd')\" -ForegroundColor Green