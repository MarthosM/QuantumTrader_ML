"""
Consolidação Automática Diária
Roda ao final do pregão para consolidar e organizar dados
"""

import os
import sys
import time
import logging
import schedule
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
import pandas as pd
import json

# Importar sistema de consolidação
from data_consolidation_system import DataConsolidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/consolidation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('AutoConsolidation')


class AutoConsolidator:
    """Sistema automático de consolidação diária"""
    
    def __init__(self):
        self.consolidator = DataConsolidator()
        
        # Horários de mercado
        self.market_close = dtime(18, 10)  # 18:10 (10min após fechamento)
        self.consolidation_time = dtime(18, 30)  # 18:30 consolidação
        self.cleanup_time = dtime(19, 0)  # 19:00 limpeza
        
        # Configurações
        self.config = {
            'auto_consolidate': True,
            'auto_cleanup': True,
            'retention_raw_days': 7,      # Manter dados brutos por 7 dias
            'retention_consolidated_days': 365,  # Manter consolidados por 1 ano
            'create_daily_report': True,
            'compress_old_files': True,
            'notification_email': None
        }
        
        # Estado
        self.last_consolidation = None
        self.stats = {
            'consolidations': 0,
            'cleanups': 0,
            'errors': 0,
            'last_run': None
        }
    
    def run_daily_consolidation(self):
        """Executa consolidação diária completa"""
        logger.info("\n" + "="*60)
        logger.info("INICIANDO CONSOLIDAÇÃO DIÁRIA AUTOMÁTICA")
        logger.info(f"Horário: {datetime.now()}")
        logger.info("="*60)
        
        try:
            # 1. Consolidar dados do dia
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            
            results = {}
            
            # Consolidar cada tipo de dado
            for data_type in ['tick', 'book', 'candles', 'predictions']:
                logger.info(f"\nConsolidando {data_type}...")
                
                # Consolidar últimas 24 horas
                output = self.consolidator.consolidate_continuous(
                    data_type=data_type,
                    start_date=datetime.combine(yesterday, dtime(9, 0)),
                    end_date=datetime.combine(today, dtime(18, 0))
                )
                
                if output:
                    results[data_type] = output
                    logger.info(f"  ✓ {data_type} consolidado: {Path(output).name}")
                else:
                    logger.warning(f"  ✗ Sem dados para {data_type}")
            
            # 2. Criar consolidação semanal se for sexta-feira
            if datetime.now().weekday() == 4:  # Sexta-feira
                self._create_weekly_consolidation()
            
            # 3. Criar consolidação mensal se for último dia útil
            if self._is_last_business_day():
                self._create_monthly_consolidation()
            
            # 4. Gerar relatório
            if self.config['create_daily_report']:
                report = self._generate_daily_report(results)
                self._save_report(report)
            
            # 5. Atualizar estatísticas
            self.stats['consolidations'] += 1
            self.stats['last_run'] = datetime.now()
            self.last_consolidation = datetime.now()
            
            logger.info("\n✅ Consolidação diária concluída com sucesso!")
            
        except Exception as e:
            logger.error(f"❌ Erro na consolidação: {e}")
            self.stats['errors'] += 1
    
    def run_cleanup(self):
        """Executa limpeza de dados antigos"""
        logger.info("\n" + "-"*60)
        logger.info("Executando limpeza de dados antigos...")
        
        try:
            # Limpar dados brutos antigos
            self.consolidator.cleanup_old_data(
                days_to_keep=self.config['retention_raw_days']
            )
            
            # Comprimir consolidados antigos
            if self.config['compress_old_files']:
                self._compress_old_consolidated()
            
            self.stats['cleanups'] += 1
            logger.info("✅ Limpeza concluída")
            
        except Exception as e:
            logger.error(f"❌ Erro na limpeza: {e}")
            self.stats['errors'] += 1
    
    def _create_weekly_consolidation(self):
        """Cria consolidação semanal"""
        logger.info("\n📊 Criando consolidação semanal...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        for data_type in ['tick', 'book', 'candles', 'predictions']:
            output = self.consolidator.consolidate_continuous(
                data_type=data_type,
                start_date=start_date,
                end_date=end_date
            )
            
            if output:
                # Renomear para indicar que é semanal
                weekly_file = Path(output).parent / f"weekly_{Path(output).name}"
                Path(output).rename(weekly_file)
                logger.info(f"  ✓ {data_type} semanal: {weekly_file.name}")
    
    def _create_monthly_consolidation(self):
        """Cria consolidação mensal"""
        logger.info("\n📊 Criando consolidação mensal...")
        
        end_date = datetime.now()
        start_date = end_date.replace(day=1)  # Primeiro dia do mês
        
        for data_type in ['tick', 'book', 'candles', 'predictions']:
            output = self.consolidator.consolidate_continuous(
                data_type=data_type,
                start_date=start_date,
                end_date=end_date
            )
            
            if output:
                # Renomear para indicar que é mensal
                month_str = end_date.strftime('%Y%m')
                monthly_file = Path(output).parent / f"monthly_{month_str}_{Path(output).name}"
                Path(output).rename(monthly_file)
                logger.info(f"  ✓ {data_type} mensal: {monthly_file.name}")
    
    def _is_last_business_day(self) -> bool:
        """Verifica se é último dia útil do mês"""
        today = datetime.now().date()
        
        # Próximo dia útil
        next_day = today + timedelta(days=1)
        while next_day.weekday() >= 5:  # Sábado ou domingo
            next_day += timedelta(days=1)
        
        # Se próximo dia útil é no próximo mês, hoje é último dia útil
        return next_day.month != today.month
    
    def _compress_old_consolidated(self):
        """Comprime arquivos consolidados antigos"""
        consolidated_path = Path('data/training/consolidated')
        if not consolidated_path.exists():
            return
        
        # Comprimir arquivos com mais de 30 dias
        cutoff = datetime.now() - timedelta(days=30)
        
        for file in consolidated_path.glob("*.parquet"):
            if file.stat().st_mtime < cutoff.timestamp():
                # Comprimir se ainda não estiver comprimido
                if not file.with_suffix('.parquet.gz').exists():
                    self.consolidator._compress_file(file)
                    logger.info(f"  Comprimido: {file.name}")
    
    def _generate_daily_report(self, results: dict) -> dict:
        """Gera relatório diário de consolidação"""
        summary = self.consolidator.get_data_summary()
        
        report = {
            'date': datetime.now().isoformat(),
            'consolidation_results': results,
            'data_summary': summary,
            'statistics': self.consolidator.stats,
            'system_stats': self.stats
        }
        
        # Adicionar métricas de qualidade
        report['quality_metrics'] = self._calculate_quality_metrics()
        
        return report
    
    def _calculate_quality_metrics(self) -> dict:
        """Calcula métricas de qualidade dos dados"""
        metrics = {
            'continuity_score': 0,
            'completeness_score': 0,
            'data_gaps': [],
            'anomalies': []
        }
        
        # Analisar dados consolidados mais recentes
        consolidated_path = Path('data/training/consolidated')
        if not consolidated_path.exists():
            return metrics
        
        recent_files = sorted(consolidated_path.glob("*.parquet"))[-5:]
        
        for file in recent_files:
            try:
                df = pd.read_parquet(file)
                
                if 'timestamp' in df.columns:
                    # Verificar continuidade
                    time_diffs = df['timestamp'].diff()
                    gaps = time_diffs[time_diffs > pd.Timedelta(minutes=5)]
                    
                    if len(gaps) > 0:
                        metrics['data_gaps'].append({
                            'file': file.name,
                            'gaps': len(gaps),
                            'max_gap': str(gaps.max())
                        })
                    
                    # Score de continuidade
                    continuity = 1 - (len(gaps) / len(df))
                    metrics['continuity_score'] = max(metrics['continuity_score'], continuity)
                
                # Score de completeness
                null_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
                metrics['completeness_score'] = 1 - null_ratio
                
            except Exception as e:
                logger.error(f"Erro analisando {file}: {e}")
        
        return metrics
    
    def _save_report(self, report: dict):
        """Salva relatório diário"""
        reports_dir = Path('data/reports')
        reports_dir.mkdir(exist_ok=True)
        
        date_str = datetime.now().strftime('%Y%m%d')
        report_file = reports_dir / f"consolidation_report_{date_str}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📋 Relatório salvo: {report_file}")
        
        # Imprimir resumo
        print("\n" + "="*60)
        print("RESUMO DO RELATÓRIO")
        print("="*60)
        print(f"Data: {report['date']}")
        print(f"Consolidações: {len(report['consolidation_results'])}")
        print(f"Qualidade:")
        print(f"  Continuidade: {report['quality_metrics']['continuity_score']:.2%}")
        print(f"  Completeness: {report['quality_metrics']['completeness_score']:.2%}")
        print(f"  Gaps detectados: {len(report['quality_metrics']['data_gaps'])}")
        print("="*60)
    
    def schedule_jobs(self):
        """Agenda jobs de consolidação"""
        # Consolidação diária às 18:30
        schedule.every().day.at("18:30").do(self.run_daily_consolidation)
        
        # Limpeza diária às 19:00
        schedule.every().day.at("19:00").do(self.run_cleanup)
        
        # Relatório de status a cada hora durante o pregão
        schedule.every().hour.between("09:00", "18:00").do(self._log_status)
        
        logger.info("📅 Jobs agendados:")
        logger.info("  - Consolidação diária: 18:30")
        logger.info("  - Limpeza: 19:00")
        logger.info("  - Status: A cada hora (9h-18h)")
    
    def _log_status(self):
        """Log de status periódico"""
        summary = self.consolidator.get_data_summary()
        
        logger.info(f"\n[STATUS] {datetime.now().strftime('%H:%M')}")
        for data_type, info in summary['data_types'].items():
            logger.info(f"  {data_type}: {info['files']} arquivos, {info['size_mb']:.1f}MB")
    
    def run_scheduler(self):
        """Executa scheduler em loop"""
        logger.info("\n🚀 Sistema de consolidação automática iniciado")
        logger.info(f"Horário atual: {datetime.now()}")
        
        self.schedule_jobs()
        
        # Loop principal
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Verificar a cada minuto
                
            except KeyboardInterrupt:
                logger.info("\n⏹️ Sistema interrompido pelo usuário")
                break
            except Exception as e:
                logger.error(f"Erro no scheduler: {e}")
                time.sleep(60)
        
        logger.info("Sistema de consolidação finalizado")


def main():
    """Executa sistema de consolidação automática"""
    
    print("\n" + "="*60)
    print("SISTEMA DE CONSOLIDAÇÃO AUTOMÁTICA")
    print("="*60)
    print("Opções:")
    print("1. Executar consolidação agora")
    print("2. Executar limpeza agora")
    print("3. Iniciar scheduler automático")
    print("4. Ver status dos dados")
    print("5. Sair")
    
    consolidator = AutoConsolidator()
    
    choice = input("\nEscolha: ").strip()
    
    if choice == '1':
        consolidator.run_daily_consolidation()
        
    elif choice == '2':
        consolidator.run_cleanup()
        
    elif choice == '3':
        print("\nIniciando scheduler automático...")
        print("CTRL+C para parar\n")
        consolidator.run_scheduler()
        
    elif choice == '4':
        consolidator._log_status()
    
    print("\nFinalizado!")


if __name__ == "__main__":
    # Criar diretório de logs
    Path('logs').mkdir(exist_ok=True)
    
    # Verificar argumentos de linha de comando
    if len(sys.argv) > 1:
        if sys.argv[1] == '--now':
            # Executar consolidação imediatamente
            consolidator = AutoConsolidator()
            consolidator.run_daily_consolidation()
        elif sys.argv[1] == '--scheduler':
            # Iniciar scheduler
            consolidator = AutoConsolidator()
            consolidator.run_scheduler()
        elif sys.argv[1] == '--cleanup':
            # Executar limpeza
            consolidator = AutoConsolidator()
            consolidator.run_cleanup()
    else:
        main()