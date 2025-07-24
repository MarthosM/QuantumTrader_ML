# Script PowerShell para iniciar ML Trading v2.0
# Ativa ambiente virtual e executa o sistema

param(
    [string]$Script = "start_ml_trading_clean.py"
)

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "    ML TRADING v2.0 - INICIALIZADOR" -ForegroundColor Cyan  
Write-Host "===============================================" -ForegroundColor Cyan

# Encontrar raiz do projeto
$projectRoot = Get-Location
if (-not (Test-Path ".env") -and -not (Test-Path ".venv")) {
    # Procurar para cima
    $current = Get-Location
    while ($current.Parent) {
        Set-Location $current.Parent
        if ((Test-Path ".env") -or (Test-Path ".venv")) {
            $projectRoot = Get-Location
            break
        }
        $current = Get-Location
    }
}

Write-Host "📁 Projeto detectado em: $projectRoot" -ForegroundColor Green
Set-Location $projectRoot

# Verificar se ambiente virtual existe
$venvPath = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "⚡ Ativando ambiente virtual..." -ForegroundColor Yellow
    
    # Ativar ambiente virtual
    & $venvPath
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Ambiente virtual ativado com sucesso" -ForegroundColor Green
        
        # Verificar versão do Python
        $pythonVersion = python --version 2>&1
        Write-Host "🐍 Python: $pythonVersion" -ForegroundColor Cyan
        
        # Procurar script para executar
        $scriptsToTry = @(
            $Script,
            "start_ml_trading_clean.py",
            "start_ml_trading_integrated.py",
            "start_ml_trading.py",
            "src\main.py"
        )
        
        $scriptFound = $false
        foreach ($scriptFile in $scriptsToTry) {
            if (Test-Path $scriptFile) {
                Write-Host "🚀 Executando: $scriptFile" -ForegroundColor Green
                python $scriptFile
                $scriptFound = $true
                break
            }
        }
        
        if (-not $scriptFound) {
            Write-Host "❌ Nenhum script de inicialização encontrado!" -ForegroundColor Red
            Write-Host "Scripts procurados:" -ForegroundColor Yellow
            foreach ($scriptFile in $scriptsToTry) {
                Write-Host "  - $scriptFile" -ForegroundColor Gray
            }
        }
        
    } else {
        Write-Host "❌ Falha ao ativar ambiente virtual" -ForegroundColor Red
    }
} else {
    Write-Host "⚠️ Ambiente virtual não encontrado em $venvPath" -ForegroundColor Yellow
    Write-Host "Executando sem ambiente virtual..." -ForegroundColor Yellow
    
    # Tentar executar mesmo assim
    if (Test-Path $Script) {
        python $Script
    } elseif (Test-Path "src\main.py") {
        Set-Location src
        python main.py
    } else {
        Write-Host "❌ Nenhum script encontrado para executar" -ForegroundColor Red
    }
}

Write-Host "`n===============================================" -ForegroundColor Cyan
Write-Host "    EXECUÇÃO FINALIZADA" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

# Pausar para ver resultado
if ($Host.Name -eq "ConsoleHost") {
    Write-Host "Pressione qualquer tecla para continuar..."
    $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
}
