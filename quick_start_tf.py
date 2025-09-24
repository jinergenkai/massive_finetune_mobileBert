#!/usr/bin/env python3
# Quick Start Script - Fine-tune MobileBERT trên MASSIVE Dataset
# Converted to TensorFlow from PyTorch

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
    print("🚀 QUICK START - MobileBERT + MASSIVE Intent Classification (TensorFlow)")
    print("="*60)
    
    # Import sau khi đã cài đặt packages
    import tensorflow as tf
    from datasets import load_dataset
    from transformers import (
        TFMobileBertForSequenceClassification,
        MobileBertTokenizer,
        create_optimizer
    )
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
    import pandas as pd
    import numpy as np
    import json
    
    # Enable mixed precision for better performance
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Cấu hình
    print(f"📊 Sử dụng ngôn ngữ: {language}")
    print(f"⏱️  Số epochs: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    print(f"💾 Output directory: {OUTPUT_DIR}")
    
    # Check GPU
    print("GPU Available:", len(tf.config.experimental.list_physical_devices('GPU')) > 0)
    if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
        print("   ⚠️ Warning: GPU not available, training may be slow!")
    
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
    print("\n3️⃣ Đang khởi tạo TensorFlow MobileBERT...")
    tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
    model = TFMobileBertForSequenceClassification.from_pretrained(
        "google/mobilebert-uncased",
        num_labels=num_labels
    )
    print("   ✓ Model initialized!")
    
    # 4. Tokenize data và tạo TensorFlow datasets
    print("\n4️⃣ Đang tokenize dữ liệu và tạo TF datasets...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['utt'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='tf'
        )
    
    # Prepare datasets
    def prepare_tf_dataset(hf_dataset, df, batch_size, is_training=False):
        # Extract texts and labels
        texts = hf_dataset['utt']
        intents = hf_dataset['intent']
        
        # Convert texts to list of strings
        texts = [str(t) for t in texts]
        
        # Create label mapping
        label_encoder_dict = dict(zip(df['intent'], df['label']))
        labels = [label_encoder_dict[intent] for intent in intents]
        
        # Tokenize all texts
        encoded = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='tf'
        )
        
        # Create TensorFlow dataset
        tf_dataset = tf.data.Dataset.from_tensor_slices({
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'token_type_ids': encoded.get('token_type_ids', tf.zeros_like(encoded['input_ids'])),  # Some tokenizers may not return token_type_ids
            'labels': tf.constant(labels)
        })
        
        if is_training:
            tf_dataset = tf_dataset.shuffle(1000)
        
        tf_dataset = tf_dataset.batch(batch_size)
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
        
        return tf_dataset

    def prepare_tf_dataset_old(hf_dataset, df, batch_size, is_training=False):
        # Tokenize
        texts = hf_dataset['utt']
        intents = hf_dataset['intent']
        
        # Create label mapping
        label_encoder_dict = dict(zip(df['intent'], df['label']))
        labels = [label_encoder_dict[intent] for intent in intents]
        
        # Tokenize all texts
        encoded = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='tf'
        )
        
        # Create TensorFlow dataset
        tf_dataset = tf.data.Dataset.from_tensor_slices({
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'token_type_ids': encoded['token_type_ids'],
            'labels': tf.constant(labels)
        })
        
        if is_training:
            tf_dataset = tf_dataset.shuffle(1000)
        
        tf_dataset = tf_dataset.batch(batch_size)
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
        
        return tf_dataset
    
    train_dataset = prepare_tf_dataset(dataset['train'], train_df, batch_size, is_training=True)
    dev_dataset = prepare_tf_dataset(dataset['validation'], dev_df, batch_size)
    test_dataset = prepare_tf_dataset(dataset['test'], test_df, batch_size)
    
    print("   ✓ TensorFlow datasets created!")
    
    # 5. Setup training
    print("\n5️⃣ Đang setup training...")
    
    # Calculate training steps
    train_steps = len(dataset['train']) // batch_size
    total_steps = train_steps * epochs
    warmup_steps = min(1000, total_steps // 10)
    
    # Create optimizer
    optimizer, schedule = create_optimizer(
        init_lr=5e-5,
        num_warmup_steps=warmup_steps,
        num_train_steps=total_steps
    )
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    print("   ✓ Training setup completed!")
    
    # 6. Setup callbacks
    print("\n6️⃣ Đang setup callbacks...")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(OUTPUT_DIR, 'best_model'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=2,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=1,
            mode='max',
            verbose=1
        )
    ]
    
    # 7. Start training
    print("\n7️⃣ Bắt đầu training...")
    print("   (Có thể mất vài phút tùy thuộc vào hardware...)")
    
    try:
        history = model.fit(
            train_dataset,
            validation_data=dev_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        print("   ✓ Training completed successfully!")
    except Exception as e:
        print(f"   ❌ Training error: {e}")
        return False
    
    # 8. Evaluate on test set
    print("\n8️⃣ Đánh giá model...")
    
    test_results = model.evaluate(test_dataset, verbose=0)
    test_accuracy = test_results[1]  # accuracy metric
    
    print(f"   ✓ Test Accuracy: {test_accuracy:.4f}")
    
    # 9. Save model
    print("\n9️⃣ Lưu model...")
    
    # Save the model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save label encoder và intent mapping
    import joblib
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, 'label_encoder.pkl'))
    
    # Save intent mapping as JSON for easy loading
    with open(os.path.join(OUTPUT_DIR, 'intent_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(intent_mapping, f, indent=2, ensure_ascii=False)
    
    # Save training history
    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history.history, f, indent=2)
    
    print(f"   ✓ Model saved to: {OUTPUT_DIR}")
    print(f"   ✓ Intent mapping saved to: {OUTPUT_DIR}/intent_mapping.json")
    
    # 10. Quick test
    print("\n🔟 Quick test với một số samples...")
    
    test_samples = [
        "set alarm for 7 am tomorrow",
        "play some music", 
        "what's the weather",
        "order pizza",
        "turn off lights"
    ]
    
    for sample in test_samples:
        inputs = tokenizer(
            sample,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='tf'
        )
        
        # Predict
        outputs = model(**inputs)
        predictions = tf.nn.softmax(outputs.logits, axis=-1)
        predicted_id = tf.argmax(predictions, axis=-1)[0].numpy()
        confidence = predictions[0][predicted_id].numpy()
        
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
    print("🔮 QUICK INFERENCE DEMO (TensorFlow)")
    print("="*50)
    
    if not os.path.exists(model_path):
        print(f"❌ Model không tìm thấy tại: {model_path}")
        print("   Hãy chạy training trước!")
        return False
    
    try:
        import tensorflow as tf
        import joblib
        import json
        from transformers import TFMobileBertForSequenceClassification, MobileBertTokenizer
        
        # Load model
        print("🔄 Đang load model...")
        model = TFMobileBertForSequenceClassification.from_pretrained(model_path)
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
                return_tensors='tf'
            )
            
            outputs = model(**inputs)
            probabilities = tf.nn.softmax(outputs.logits, axis=-1)
            
            # Top 3 predictions
            top_probs, top_indices = tf.nn.top_k(probabilities[0], k=3)
            
            print(f"\n📊 Kết quả cho: '{text}'")
            print("-" * 50)
            for i in range(3):
                index = top_indices[i].numpy()
                intent_text = intent_mapping.get(index, f"Unknown_ID_{index}")
                confidence = top_probs[i].numpy()
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
    print("🤖 MobileBERT + MASSIVE Dataset Fine-tuning (TensorFlow)")
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