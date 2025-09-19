def check_torch_cuda():
    import torch
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


def check_and_install_requirements():
    """Kiểm tra và cài đặt các thư viện cần thiết"""
    required_packages = [
        'transformers==4.46.1',
        'torch',
        'datasets', 
        'scikit-learn',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'joblib',
        'accelerate'
    ]
    
    try:
        import transformers
        import torch
        import datasets
        import sklearn
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import joblib
        print("✓ Tất cả packages đã được cài đặt!")
        return True
    except ImportError as e:
        print(f"❌ Thiếu package: {e}")
        print("Đang cài đặt packages cần thiết...")
        
        for package in required_packages:
            os.system(f"pip install {package}")
        
        return True

def show_intent_list(model_path="./models/mobilebert-massive-quick"):
    """Hiển thị danh sách tất cả intents"""
    print("\n" + "="*50)
    print("📋 DANH SÁCH TẤT CẢ INTENTS")
    print("="*50)
    
    try:
        import json
        import joblib
        
        # Load intent mapping
        intent_mapping_path = os.path.join(model_path, 'intent_mapping.json')
        if os.path.exists(intent_mapping_path):
            with open(intent_mapping_path, 'r', encoding='utf-8') as f:
                intent_mapping = json.load(f)
                intent_mapping = {int(k): v for k, v in intent_mapping.items()}
        else:
            # Fallback
            label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
            intent_mapping = {idx: intent for idx, intent in enumerate(label_encoder.classes_)}
        
        print(f"Tổng cộng: {len(intent_mapping)} intents\n")
        
        # Group by category if possible
        intents_by_category = {}
        for idx, intent in intent_mapping.items():
            if '_' in intent:
                category = intent.split('_')[0]
            else:
                category = 'other'
            
            if category not in intents_by_category:
                intents_by_category[category] = []
            intents_by_category[category].append((idx, intent))
        
        # Print by category
        for category, intents in sorted(intents_by_category.items()):
            print(f"🔸 {category.upper()} ({len(intents)} intents):")
            for idx, intent in sorted(intents, key=lambda x: x[1]):
                print(f"   {idx:3d}: {intent}")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")
