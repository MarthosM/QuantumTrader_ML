@echo off
echo ========================================
echo  QUANTUM TRADER ML - SISTEMA PRODUCAO
echo  65 Features + HMARL + Data Recording
echo ========================================
echo.
echo IMPORTANTE:
echo - O sistema abrira o monitor em nova janela
echo - Para funcionar precisa do Profit Chart aberto
echo - Se o mercado esta fechado, nao havera dados
echo.
pause

REM Ativar ambiente virtual
call .venv\Scripts\activate

REM Iniciar sistema
python start_production_65features.py

pause