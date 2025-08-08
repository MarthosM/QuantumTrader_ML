"""
Script automatizado para consolidar dados do Book Collector
Executa sem necessidade de interação do usuário
"""

from consolidate_book_data import consolidate_auto
from datetime import datetime
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def main():
    """Executa consolidação automática"""
    
    # Verificar se foi passada uma data específica
    date = None
    if len(sys.argv) > 1:
        date = sys.argv[1]
        print(f"Consolidando dados da data: {date}")
    else:
        date = datetime.now().strftime('%Y%m%d')
        print(f"Consolidando dados de hoje: {date}")
    
    # Executar consolidação automática
    try:
        consolidate_auto(date, consolidate_all=True)
        print("\n[OK] Consolidacao automatica finalizada com sucesso!")
        
    except Exception as e:
        print(f"\n[ERRO] Erro durante consolidacao: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())