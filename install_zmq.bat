@echo off
echo Instalando pyzmq no ambiente virtual...
call .venv\Scripts\activate
python -m pip install pyzmq
echo pyzmq instalado com sucesso!
pause