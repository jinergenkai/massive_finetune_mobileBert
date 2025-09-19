def print_help():
    """In hướng dẫn sử dụng"""
    print("""
📚 HƯỚNG DẪN SỬ DỤNG:

🏋️ Training:
   python quick_start.py train [language] [epochs]
   
   Ví dụ:
   python quick_start.py train en-US 3
   python quick_start.py train vi-VN 2

🔮 Inference:
   python quick_start.py inference [model_path]
   
   Ví dụ:
   python quick_start.py inference ./models/mobilebert-massive-quick

📖 Help:
   python quick_start.py help

🚀 Quick start (default):
   python quick_start.py
   (Sẽ training với en-US, 2 epochs, sau đó demo inference)

🌍 Supported languages:
   en-US, vi-VN, zh-CN, fr-FR, de-DE, es-ES, ja-JP, ko-KR, 
   hi-IN, th-TH, id-ID, ms-MY, tl-PH, và 40+ ngôn ngữ khác...
""")
