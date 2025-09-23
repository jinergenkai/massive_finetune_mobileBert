#!/usr/bin/env python3
# Quick Start Script - Fine-tune MobileBERT trên MASSIVE Dataset
# Fixed: Correct intent label mapping

import os
import sys

from src.instruction import print_help
from src.util import show_intent_list, check_and_install_requirements
from src.data_processing import prepare_data

OUTPUT_DIR = "./models/mobilebert-massive-quick"
MAX_LENGTH = 128

def quick_train(language="en-US", epochs=2, batch_size=16):
    """
    Training nhanh với cấu hình tối ưu
    
    Args:
        language: Ngôn ngữ dataset (mặc định: en-US)
        epochs: Số epochs (mặc định: 2 để train nhanh)
        batch_size: Batch size (mặc định: 16)
    """
    print("="*60)
    print("🚀 QUICK START - MobileBERT + MASSIVE Intent Classification")
    print("="*60)
    
    # Import sau khi đã cài đặt packages
    import torch
    from datasets import load_dataset
    from transformers import (
        MobileBertTokenizer, 
        MobileBertForSequenceClassification,
        TrainingArguments, 
        Trainer,
        DataCollatorWithPadding
    )
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
    import pandas as pd
    import numpy as np
    import json
    
    # Cấu hình
    
    print(f"📊 Sử dụng ngôn ngữ: {language}")
    print(f"⏱️  Số epochs: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    print(f"💾 Output directory: {OUTPUT_DIR}")
    
    # Tạo thư mục output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load dataset
    print("\n1️⃣ Đang load MASSIVE dataset...")
    try:
        dataset = load_dataset("AmazonScience/massive", language)
        print(f"   ✓ Loaded successfully! Train: {len(dataset['train'])}, "
              f"Dev: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        return False
    
    # 2. Prepare data và tạo intent mapping
    print("\n2️⃣ Đang chuẩn bị dữ liệu...")
    train_df, dev_df, test_df, label_encoder = prepare_data(dataset)
    num_labels = len(label_encoder.classes_)
    
    # Lấy text intent từ ClassLabel của dataset
    intent_texts = dataset['train'].features['intent'].names  # danh sách text intent

    # Tạo mapping
    intent_mapping = {idx: intent_text for idx, intent_text in enumerate(intent_texts)}

    print(f"   ✓ Số intent classes: {len(intent_mapping)}")
    print(f"   ✓ Created intent mapping: {len(intent_mapping)} intents")
    print(f"   ✓ Sample intents: {list(intent_mapping.items())[:5]}")
    
    # 3. Initialize tokenizer and model
    print("\n3️⃣ Đang khởi tạo MobileBERT...")
    tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
    model = MobileBertForSequenceClassification.from_pretrained(
        "google/mobilebert-uncased",
        num_labels=num_labels
    )
    print("   ✓ Model initialized!")
    
    # 4. Tokenize data
    print("\n4️⃣ Đang tokenize dữ liệu...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['utt'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        )
    
    # Apply tokenization
    train_dataset = dataset['train'].map(tokenize_function, batched=True)
    dev_dataset = dataset['validation'].map(tokenize_function, batched=True)
    test_dataset = dataset['test'].map(tokenize_function, batched=True)
    
    # Add labels
    def add_labels(examples, df):
        label_encoder_dict = dict(zip(df['intent'], df['label']))
        examples['labels'] = [label_encoder_dict[intent] for intent in examples['intent']]
        return examples
    
    train_dataset = train_dataset.map(lambda x: add_labels(x, train_df), batched=True)
    dev_dataset = dev_dataset.map(lambda x: add_labels(x, dev_df), batched=True)
    test_dataset = test_dataset.map(lambda x: add_labels(x, test_df), batched=True)
    
    print("   ✓ Tokenization completed!")
    
    # 5. Setup training
    print("\n5️⃣ Đang setup training...")

    print("CUDA available:", torch.cuda.is_available())
    print("Model device:", next(model.parameters()).device)
    if not torch.cuda.is_available():
        print("   ⚠️ Warning: CUDA not available, training may be slow!")
        return
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy}
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=5e-5,
        logging_dir=f'{OUTPUT_DIR}/logs',
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        report_to=None,
        dataloader_num_workers=2,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("   ✓ Training setup completed!")
    
    # 6. Start training
    print("\n6️⃣ Bắt đầu training...")
    print("   (Có thể mất vài phút tùy thuộc vào hardware...)")
    
    try:
        trainer.train()
        print("   ✓ Training completed successfully!")
    except Exception as e:
        print(f"   ❌ Training error: {e}")
        return False
    
    # 7. Evaluate
    print("\n7️⃣ Đánh giá model...")
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    test_accuracy = test_results['eval_accuracy']
    
    print(f"   ✓ Test Accuracy: {test_accuracy:.4f}")
    
    # 8. Save model
    print("\n8️⃣ Lưu model...")
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save label encoder và intent mapping
    import joblib
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, 'label_encoder.pkl'))
    
    # Save intent mapping as JSON for easy loading
    with open(os.path.join(OUTPUT_DIR, 'intent_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(intent_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Model saved to: {OUTPUT_DIR}")
    print(f"   ✓ Intent mapping saved to: {OUTPUT_DIR}/intent_mapping.json")
    
    # 9. Quick test
    print("\n9️⃣ Quick test với một số samples...")
    
    test_samples = [
        "set alarm for 7 am tomorrow",
        "play some music", 
        "what's the weather",
        "order pizza",
        "turn off lights"
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    for sample in test_samples:
        inputs = tokenizer(
            sample,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_id = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_id].item()
        
        # Use intent mapping to get text label
        predicted_intent = intent_mapping.get(predicted_id, f"Unknown_ID_{predicted_id}")
        
        print(f"   📝 '{sample}' -> {predicted_intent} ({confidence:.3f}) | Label ID: {predicted_id}")
    
    print("\n" + "="*60)
    print("🎉 HOÀN THÀNH!")
    print(f"✅ Model đã được train và lưu tại: {OUTPUT_DIR}")
    print(f"✅ Test accuracy: {test_accuracy:.4f}")
    print("✅ Có thể sử dụng model cho inference!")
    print("="*60)
    
    return True

def quick_inference_demo(model_path=OUTPUT_DIR):
    """Demo nhanh inference"""
    print("\n" + "="*50)
    print("🔮 QUICK INFERENCE DEMO")
    print("="*50)
    
    if not os.path.exists(model_path):
        print(f"❌ Model không tìm thấy tại: {model_path}")
        print("   Hãy chạy training trước!")
        return False
    
    try:
        import torch
        import joblib
        import json
        from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
        
        # Load model
        print("🔄 Đang load model...")
        model = MobileBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = MobileBertTokenizer.from_pretrained(model_path)
        
        # Load intent mapping
        intent_mapping_path = os.path.join(model_path, 'intent_mapping.json')
        if os.path.exists(intent_mapping_path):
            with open(intent_mapping_path, 'r', encoding='utf-8') as f:
                intent_mapping = json.load(f)
                # Convert string keys to int
                intent_mapping = {int(k): v for k, v in intent_mapping.items()}
            print(f"   ✅ Loaded intent mapping: {len(intent_mapping)} intents")
        else:
            # Fallback to label encoder (old method)
            label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
            intent_mapping = {idx: intent for idx, intent in enumerate(label_encoder.classes_)}
            print("   ⚠️ Using fallback label encoder mapping")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        print("✅ Model loaded successfully!")
        
        # Show some example intents
        print(f"\n📋 Available intents (showing first 10):")
        for i, intent in list(intent_mapping.items())[:10]:
            print(f"   {i}: {intent}")
        if len(intent_mapping) > 10:
            print(f"   ... và {len(intent_mapping)-10} intents khác")
        
        # Interactive demo
        print("\n💬 Interactive mode (gõ 'quit' để thoát):")
        
        while True:
            text = input("\n🎯 Nhập câu cần phân loại: ").strip()
            
            if text.lower() == 'quit':
                break
                
            if not text:
                continue
            
            # Predict
            inputs = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Top 3 predictions
                top_probs, top_indices = torch.topk(probabilities[0], 3)
                
                print(f"\n📊 Kết quả cho: '{text}'")
                print("-" * 50)
                for i in range(3):
                    index = top_indices[i].item()
                    intent_text = intent_mapping.get(index, f"Unknown_ID_{index}")
                    confidence = top_probs[i].item()
                    print(f"   {i+1}. {intent_text} ({confidence:.4f}) [ID: {index}]")
        
        print("👋 Cảm ơn bạn đã sử dụng!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🤖 MobileBERT + MASSIVE Dataset Fine-tuning")
    print("=" * 60)
    
    # Parse arguments
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "train":
            # Training mode
            language = sys.argv[2] if len(sys.argv) > 2 else "en-US"
            epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
            
            print(f"🏋️ Training mode - Language: {language}, Epochs: {epochs}")
            
            # Check requirements
            if check_and_install_requirements():
                quick_train(language=language, epochs=epochs)
            
        elif command == "inference":
            # Inference mode
            model_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
            quick_inference_demo(model_path)
            
        elif command == "intents":
            # Show intent list
            model_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
            show_intent_list(model_path)
            
        elif command == "help":
            print_help()
            
        else:
            print(f"❌ Unknown command: {command}")
            print_help()
    else:
        # Default: training
        print("🏋️ Default mode: Training")
        
        # Check requirements
        if check_and_install_requirements():
            success = quick_train()
            
            if success:
                print("\n🎯 Bạn có muốn test inference không? (y/n): ", end="")
                choice = input().strip().lower()
                
                if choice == 'y':
                    quick_inference_demo()

if __name__ == "__main__":
    main()