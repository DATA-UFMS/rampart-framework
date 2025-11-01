#!/usr/bin/env python3
"""
Script para limpar todos os arquivos gerados automaticamente no repositório.

Este script remove:
- Arquivos de dados (.parquet, .csv, .json, .duckdb, .h5, .hdf5, .xlsx)
- Logs (.log, arquivos em pastas logs/)
- Cache Python (__pycache__/, *.pyc, *.pyo)
- Arquivos temporários (tmp_*, temp_*, *_backup*, *_old*)
- Pastas de output (outputs/, benchmark_results/, validation_results/, etc.)
- Cache de ferramentas (dask-worker-space/, .dask/, .pandas_cache/, etc.)
- Arquivos de profiling (*.prof, *.lprof)
- Arquivos do sistema (.DS_Store, Thumbs.db, etc.)

USO:
    python cleanup_generated_files.py [--dry-run] [--verbose]
    
    --dry-run: Mostra o que seria removido sem remover
    --verbose: Mostra detalhes da operação
"""

import os
import shutil
import glob
import argparse
from pathlib import Path
import sys

class RepositoryCleanup:
    def __init__(self, dry_run=False, verbose=False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.removed_files = 0
        self.removed_dirs = 0
        self.total_size = 0
        
        # Extensões de arquivos para remover
        self.file_extensions = [
            '*.log', '*.json', '*.parquet', '*.csv', '*.duckdb',
            '*.h5', '*.hdf5', '*.xlsx', '*.pkl', '*.pickle',
            '*.joblib', '*.model', '*.bin', '*.prof', '*.lprof',
            '*.pyc', '*.pyo', '*.pyd', '*.so', '*.egg-info',
            '*.tmp', '*.temp', '*.backup', '*.swp', '*.swo',
            '*.DS_Store', 'Thumbs.db', 'ehthumbs.db'
        ]
        
        # Padrões de nomes de arquivos para remover
        self.file_patterns = [
            'tmp_*', 'temp_*', '*_backup*', '*_old*',
            '*_log.txt', '*_results.json', 'pipeline_results_*.json',
            '.DS_Store*', '._*', '*~'
        ]
        
        # Diretórios para remover completamente
        self.directories_to_remove = [
            'outputs', 'logs', 'data', 'benchmark_results',
            'validation_results', 'equivalence_results',
            'dask-worker-space', '.dask', '.pandas_cache',
            'sklearn_cache', 'test_output', '.ipynb_checkpoints',
            'backups_before_output_update'
        ]
        
        # Diretórios de cache Python
        self.python_cache_dirs = ['__pycache__']
        
        # Arquivos específicos para preservar (mesmo que coincidam com padrões)
        self.preserve_files = [
            'README.md', 'requirements.txt', '.gitignore',
            'setup.py', 'pyproject.toml', 'Pipfile'
        ]

    def get_file_size(self, path):
        """Retorna o tamanho do arquivo em bytes."""
        try:
            return os.path.getsize(path)
        except (OSError, IOError):
            return 0

    def get_dir_size(self, path):
        """Retorna o tamanho total de um diretório em bytes."""
        total = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total += os.path.getsize(filepath)
                    except (OSError, IOError):
                        continue
        except (OSError, IOError):
            pass
        return total

    def format_size(self, size_bytes):
        """Formata tamanho em bytes para formato legível."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def should_preserve_file(self, filepath):
        """Verifica se um arquivo deve ser preservado."""
        filename = os.path.basename(filepath)
        return filename in self.preserve_files

    def remove_file(self, filepath):
        """Remove um arquivo individual."""
        if self.should_preserve_file(filepath):
            if self.verbose:
                print(f"  PRESERVANDO: {filepath}")
            return False
            
        size = self.get_file_size(filepath)
        
        if self.dry_run:
            print(f"  REMOVERIA: {filepath} ({self.format_size(size)})")
            self.total_size += size
            self.removed_files += 1
            return True
        else:
            try:
                os.remove(filepath)
                if self.verbose:
                    print(f"  REMOVIDO: {filepath} ({self.format_size(size)})")
                self.total_size += size
                self.removed_files += 1
                return True
            except (OSError, IOError) as e:
                if self.verbose:
                    print(f"  ERRO ao remover {filepath}: {e}")
                return False

    def remove_directory(self, dirpath):
        """Remove um diretório e todo seu conteúdo."""
        if not os.path.exists(dirpath):
            return False
            
        size = self.get_dir_size(dirpath)
        
        if self.dry_run:
            print(f"  REMOVERIA DIRETÓRIO: {dirpath} ({self.format_size(size)})")
            self.total_size += size
            self.removed_dirs += 1
            return True
        else:
            try:
                shutil.rmtree(dirpath)
                if self.verbose:
                    print(f"  REMOVIDO DIRETÓRIO: {dirpath} ({self.format_size(size)})")
                self.total_size += size
                self.removed_dirs += 1
                return True
            except (OSError, IOError) as e:
                if self.verbose:
                    print(f"  ERRO ao remover diretório {dirpath}: {e}")
                return False

    def clean_by_extensions(self):
        """Remove arquivos por extensão."""
        print(" Limpando arquivos por extensão...")
        
        for pattern in self.file_extensions:
            files = glob.glob(f"**/{pattern}", recursive=True)
            if files:
                print(f"  Padrão: {pattern}")
                for filepath in files:
                    self.remove_file(filepath)

    def clean_by_patterns(self):
        """Remove arquivos por padrões de nome."""
        print("\n Limpando arquivos por padrões de nome...")
        
        for pattern in self.file_patterns:
            files = glob.glob(f"**/{pattern}", recursive=True)
            if files:
                print(f"  Padrão: {pattern}")
                for filepath in files:
                    if os.path.isfile(filepath):
                        self.remove_file(filepath)

    def clean_directories(self):
        """Remove diretórios específicos."""
        print("\n Limpando diretórios...")
        
        for dirname in self.directories_to_remove:
            # Busca em todos os níveis
            dirs = glob.glob(f"**/{dirname}", recursive=True)
            for dirpath in dirs:
                if os.path.isdir(dirpath):
                    print(f"  Diretório: {dirname}")
                    self.remove_directory(dirpath)

    def clean_python_cache(self):
        """Remove cache Python."""
        print("\n Limpando cache Python...")
        
        for dirname in self.python_cache_dirs:
            dirs = glob.glob(f"**/{dirname}", recursive=True)
            for dirpath in dirs:
                if os.path.isdir(dirpath):
                    print(f"  Cache Python: {dirpath}")
                    self.remove_directory(dirpath)

    def clean_empty_directories(self):
        """Remove diretórios vazios."""
        print("\n Removendo diretórios vazios...")
        
        # Busca diretórios vazios (executa múltiplas vezes para pegar diretórios aninhados)
        for _ in range(5):  # Máximo 5 níveis de aninhamento
            empty_dirs = []
            for root, dirs, files in os.walk('.'):
                if not dirs and not files and root != '.':
                    empty_dirs.append(root)
            
            if not empty_dirs:
                break
                
            for dirpath in empty_dirs:
                if self.dry_run:
                    print(f"  REMOVERIA DIRETÓRIO VAZIO: {dirpath}")
                    self.removed_dirs += 1
                else:
                    try:
                        os.rmdir(dirpath)
                        if self.verbose:
                            print(f"  REMOVIDO DIRETÓRIO VAZIO: {dirpath}")
                        self.removed_dirs += 1
                    except (OSError, IOError):
                        pass

    def run_cleanup(self):
        """Executa a limpeza completa."""
        print(" Iniciando limpeza do repositório...")
        print(f"Modo: {'DRY RUN (simulação)' if self.dry_run else 'EXECUÇÃO REAL'}")
        print("=" * 60)
        
        # Executa todas as etapas de limpeza
        self.clean_by_extensions()
        self.clean_by_patterns()
        self.clean_directories()
        self.clean_python_cache()
        self.clean_empty_directories()
        
        # Relatório final
        print("\n" + "=" * 60)
        print("⚙ RELATÓRIO FINAL:")
        print(f"  Arquivos removidos: {self.removed_files}")
        print(f"  Diretórios removidos: {self.removed_dirs}")
        print(f"  Espaço liberado: {self.format_size(self.total_size)}")
        
        if self.dry_run:
            print("\n SIMULAÇÃO - Nenhum arquivo foi realmente removido!")
            print("   Execute sem --dry-run para realizar a limpeza.")
        else:
            print("\n Limpeza concluída!")

def main():
    parser = argparse.ArgumentParser(
        description="Limpa arquivos gerados automaticamente no repositório",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Simula a limpeza sem remover arquivos'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true', 
        help='Mostra detalhes da operação'
    )
    
    args = parser.parse_args()
    
    # Confirma se não é dry-run
    if not args.dry_run:
        print("  ATENÇÃO: Esta operação irá REMOVER arquivos permanentemente!")
        print("   Arquivos de dados, logs, cache e outputs serão deletados.")
        response = input("   Deseja continuar? (digite 'sim' para confirmar): ")
        
        if response.lower() not in ['sim', 'yes', 's', 'y']:
            print(" Operação cancelada.")
            sys.exit(0)
    
    # Executa limpeza
    cleanup = RepositoryCleanup(dry_run=args.dry_run, verbose=args.verbose)
    cleanup.run_cleanup()

if __name__ == "__main__":
    main()