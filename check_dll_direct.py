"""
Verifica a DLL diretamente usando ferramentas do Windows
"""

import subprocess
import os
import sys
from pathlib import Path
import ctypes
from ctypes import wintypes

def check_with_dumpbin():
    """Tenta usar dumpbin se disponível"""
    print("\n" + "="*60)
    print("VERIFICANDO COM DUMPBIN")
    print("="*60)
    
    dll_path = Path('./ProfitDLL64.dll').absolute()
    
    # Procurar dumpbin no Visual Studio
    vs_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC"
    ]
    
    dumpbin_found = False
    for vs_path in vs_paths:
        if os.path.exists(vs_path):
            # Procurar dumpbin em subdiretórios
            for root, dirs, files in os.walk(vs_path):
                if 'dumpbin.exe' in files:
                    dumpbin_path = os.path.join(root, 'dumpbin.exe')
                    print(f"Dumpbin encontrado: {dumpbin_path}")
                    
                    try:
                        # Executar dumpbin /exports
                        result = subprocess.run(
                            [dumpbin_path, '/exports', str(dll_path)],
                            capture_output=True,
                            text=True
                        )
                        
                        if result.returncode == 0:
                            print("\nExports encontrados:")
                            print("-" * 60)
                            lines = result.stdout.split('\n')
                            
                            in_exports = False
                            exports = []
                            
                            for line in lines:
                                if 'ordinal' in line and 'name' in line:
                                    in_exports = True
                                    continue
                                    
                                if in_exports and line.strip():
                                    parts = line.split()
                                    if len(parts) >= 2 and parts[0].isdigit():
                                        # Extrair nome da função
                                        func_name = ' '.join(parts[2:]) if len(parts) > 2 else parts[1]
                                        exports.append(func_name)
                                        
                            if exports:
                                print(f"Total de exports: {len(exports)}")
                                print("\nPrimeiros 20 exports:")
                                for exp in exports[:20]:
                                    print(f"  - {exp}")
                            else:
                                print("Nenhum export encontrado")
                                
                        dumpbin_found = True
                        break
                    except Exception as e:
                        print(f"Erro ao executar dumpbin: {e}")
                        
            if dumpbin_found:
                break
                
    if not dumpbin_found:
        print("Dumpbin não encontrado. Instale o Visual Studio Build Tools.")
        

def check_dll_manually():
    """Verifica a DLL manualmente usando Windows API"""
    print("\n" + "="*60)
    print("VERIFICAÇÃO MANUAL COM WINDOWS API")
    print("="*60)
    
    dll_path = Path('./ProfitDLL64.dll').absolute()
    
    # Usar kernel32 diretamente
    kernel32 = ctypes.windll.kernel32
    
    # LoadLibraryEx com diferentes flags
    LOAD_LIBRARY_AS_DATAFILE = 0x00000002
    DONT_RESOLVE_DLL_REFERENCES = 0x00000001
    
    print("\n1. Tentando LoadLibraryEx com DONT_RESOLVE_DLL_REFERENCES...")
    handle = kernel32.LoadLibraryExW(
        ctypes.c_wchar_p(str(dll_path)),
        None,
        DONT_RESOLVE_DLL_REFERENCES
    )
    
    if handle:
        print(f"   ✓ Handle obtido: {hex(handle)}")
        
        # Tentar enumerar algumas funções conhecidas
        print("\n   Procurando funções conhecidas:")
        
        test_functions = [
            # Variações de DLLInitialize
            'DLLInitialize', '_DLLInitialize', 'DLLInitialize@0',
            'DLLInitializeLogin', '_DLLInitializeLogin', 'DLLInitializeLogin@12',
            
            # Variações de callbacks
            'SetStateCallback', '_SetStateCallback', 'SetStateCallback@4',
            'SetOfferBookCallback', '_SetOfferBookCallback',
            'SetPriceBookCallback', '_SetPriceBookCallback',
            
            # Variações de subscribe
            'SubscribeTicker', '_SubscribeTicker', 'SubscribeTicker@4',
            
            # Possíveis exports C++
            '?DLLInitialize@@YAHXZ',
            '?DLLInitializeLogin@@YAHPAD00@Z',
            '?SetStateCallback@@YAXP6AXH@Z@Z'
        ]
        
        found_count = 0
        for func in test_functions:
            addr = kernel32.GetProcAddress(handle, func.encode())
            if addr:
                print(f"   ✓ {func}: {hex(addr)}")
                found_count += 1
                
        if found_count == 0:
            print("   ⚠ Nenhuma função conhecida encontrada")
            
        kernel32.FreeLibrary(handle)
    else:
        error = kernel32.GetLastError()
        print(f"   ✗ Falha ao carregar: Erro {error}")
        

def test_profitdll_directly():
    """Testa a DLL diretamente com base no manual"""
    print("\n" + "="*60)
    print("TESTE DIRETO BASEADO NO MANUAL")
    print("="*60)
    
    dll_path = Path('./ProfitDLL64.dll').absolute()
    
    # Verificar se existe arquivo .lib ou .def junto
    dll_dir = dll_path.parent
    related_files = [
        'ProfitDLL64.lib',
        'ProfitDLL64.def',
        'ProfitDLL.lib',
        'ProfitDLL.def',
        'ProfitDLL64.h',
        'ProfitDLL.h'
    ]
    
    print("\nArquivos relacionados:")
    for file in related_files:
        file_path = dll_dir / file
        if file_path.exists():
            print(f"  ✓ {file} encontrado")
            
            # Se for .def, ler conteúdo
            if file.endswith('.def'):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    print(f"\n    Conteúdo de {file}:")
                    print("    " + "-"*40)
                    lines = content.split('\n')[:20]  # Primeiras 20 linhas
                    for line in lines:
                        print(f"    {line}")
                except:
                    pass
        else:
            print(f"  ✗ {file} não encontrado")
            
    # Tentar carregar com ctypes.cdll (C calling convention)
    print("\n\nTestando com CDLL (convenção C):")
    try:
        dll = ctypes.CDLL(str(dll_path))
        
        # Verificar __dict__
        print(f"Atributos no __dict__: {len(dll.__dict__)}")
        
        # Tentar acessar função diretamente
        try:
            # Assumir que DLLInitializeLogin existe
            dll.DLLInitializeLogin.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
            dll.DLLInitializeLogin.restype = ctypes.c_int
            
            result = dll.DLLInitializeLogin(b'HMARL', b'29936354842', b'Ultrajiu33!')
            print(f"✓ DLLInitializeLogin executado: {result}")
        except AttributeError:
            print("✗ DLLInitializeLogin não encontrado como atributo")
            
            # Tentar com getattr
            try:
                func = getattr(dll, 'DLLInitializeLogin')
                print("✓ Encontrado via getattr")
            except:
                print("✗ Não encontrado via getattr")
                
    except Exception as e:
        print(f"✗ Erro com CDLL: {e}")
        

def check_system_info():
    """Verifica informações do sistema"""
    print("\n" + "="*60)
    print("INFORMAÇÕES DO SISTEMA")
    print("="*60)
    
    import platform
    
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # Verificar se estamos em 64-bit
    print(f"\nPython bits: {sys.maxsize.bit_length() + 1}")
    print(f"Is 64-bit OS: {platform.machine().endswith('64')}")
    
    # Verificar Visual C++ Redistributables
    print("\n\nVisual C++ Redistributables:")
    
    # Caminhos comuns
    vc_paths = [
        r"C:\Windows\System32\msvcp140.dll",
        r"C:\Windows\System32\vcruntime140.dll",
        r"C:\Windows\System32\vcruntime140_1.dll",
        r"C:\Windows\System32\msvcp120.dll",
        r"C:\Windows\System32\msvcr120.dll"
    ]
    
    for vc_path in vc_paths:
        if os.path.exists(vc_path):
            print(f"  ✓ {os.path.basename(vc_path)}")
        else:
            print(f"  ✗ {os.path.basename(vc_path)}")
            

def suggest_solutions():
    """Sugere soluções possíveis"""
    print("\n" + "="*60)
    print("POSSÍVEIS SOLUÇÕES")
    print("="*60)
    
    print("\n1. Verificar com o suporte:")
    print("   - A DLL fornecida está correta?")
    print("   - Existe um arquivo .h (header) com as declarações?")
    print("   - Existe um arquivo .lib ou .def?")
    print("   - Qual a convenção de chamada correta?")
    
    print("\n2. Instalar dependências:")
    print("   - Visual C++ Redistributable 2015-2022")
    print("   - Visual C++ Redistributable 2013")
    print("   - Visual C++ Redistributable 2012")
    
    print("\n3. Testar com ferramentas:")
    print("   - Dependency Walker (depends.exe)")
    print("   - Process Monitor para ver acessos à DLL")
    print("   - API Monitor para interceptar chamadas")
    
    print("\n4. Alternativas:")
    print("   - Solicitar versão com exports explícitos")
    print("   - Solicitar documentação técnica da DLL")
    print("   - Verificar se há SDK oficial")
    

def main():
    print("\n" + "="*60)
    print("VERIFICAÇÃO COMPLETA DA PROFITDLL")
    print("="*60)
    
    # 1. Informações do sistema
    check_system_info()
    
    # 2. Verificar DLL manualmente
    check_dll_manually()
    
    # 3. Testar diretamente
    test_profitdll_directly()
    
    # 4. Tentar dumpbin
    check_with_dumpbin()
    
    # 5. Sugestões
    suggest_solutions()
    

if __name__ == "__main__":
    main()