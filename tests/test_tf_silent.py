"""
Verificador silencioso de funcionalidade do TensorFlow
Testa todas as funcionalidades sem prints que causam travamento
"""
import sys
import os
sys.path.append('src')

def test_tensorflow_silent():
    """Testa TensorFlow de forma silenciosa"""
    results = {
        'tensorflow_available': False,
        'keras_available': False,
        'classes_functional': False,
        'model_creation': False,
        'compilation': False,
        'version': None,
        'error': None
    }
    
    try:
        # Teste 1: Importar TensorFlow
        import tensorflow as tf
        results['tensorflow_available'] = True
        results['version'] = tf.__version__
        
        # Teste 2: Verificar keras
        if hasattr(tf, 'keras'):
            keras = tf.keras
            results['keras_available'] = True
        else:
            import keras
            results['keras_available'] = True
        
        # Teste 3: Verificar classes
        Sequential = keras.Sequential
        LSTM = keras.layers.LSTM
        Dense = keras.layers.Dense
        Adam = keras.optimizers.Adam
        results['classes_functional'] = True
        
        # Teste 4: Criar modelo simples
        model = Sequential([
            LSTM(32, input_shape=(10, 5)),
            Dense(3, activation='softmax')
        ])
        results['model_creation'] = True
        
        # Teste 5: Compilar modelo
        model.compile(optimizer=Adam(), loss='categorical_crossentropy')
        results['compilation'] = True
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def test_model_manager():
    """Testa ModelManager silenciosamente"""
    results = {
        'import_success': False,
        'tf_available_flag': False,
        'ensemble_creation': False,
        'error': None
    }
    
    try:
        from model_manager import ModelManager, TF_AVAILABLE
        results['import_success'] = True
        results['tf_available_flag'] = TF_AVAILABLE
        
        # Testar criação de ensemble
        if TF_AVAILABLE:
            mm = ModelManager()
            ensemble_init = mm.initialize_ensemble(n_features=10)
            results['ensemble_creation'] = ensemble_init
            
    except Exception as e:
        results['error'] = str(e)
    
    return results

if __name__ == "__main__":
    # Teste TensorFlow direto
    tf_results = test_tensorflow_silent()
    
    # Teste ModelManager
    mm_results = test_model_manager()
    
    # Status final
    all_good = (
        tf_results['tensorflow_available'] and
        tf_results['keras_available'] and
        tf_results['classes_functional'] and
        tf_results['model_creation'] and
        tf_results['compilation'] and
        mm_results['import_success'] and
        mm_results['tf_available_flag']
    )
    
    # Saída mínima
    if all_good:
        print("TENSORFLOW_STATUS:OK")
        print(f"VERSION:{tf_results['version']}")
        print(f"ENSEMBLE_OK:{mm_results['ensemble_creation']}")
    else:
        print("TENSORFLOW_STATUS:ERROR")
        if tf_results['error']:
            print(f"TF_ERROR:{tf_results['error']}")
        if mm_results['error']:
            print(f"MM_ERROR:{mm_results['error']}")
    
    # Exit code
    exit(0 if all_good else 1)
