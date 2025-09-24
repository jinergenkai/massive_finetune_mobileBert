#!/usr/bin/env python3
# Quick Start Script - Fine-tune MobileBERT trên MASSIVE Dataset
# Fixed version for TensorFlow 2.10

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
    
    # DISABLE mixed precision - gây ra NaN loss với TF 2.10
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Cấu hình
    print(f"📊 Sử dụng ngôn ngữ: {language}")
    print(f"⏱️  Số epochs: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    print(f"💾 Output directory: {OUTPUT_DIR}")
    
    # Check GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid OOM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU Available: {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("GPU Available: False")
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
    intent_texts = dataset['train'].features['intent'].names
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
    
    def prepare_tf_dataset(hf_dataset, df, batch_size, is_training=False):
        """Fixed version of dataset preparation"""
        # Extract texts and labels
        texts = [str(t) for t in hf_dataset['utt']]
        intents = hf_dataset['intent']
        
        # Create label mapping - FIXED: use proper mapping
        intent_to_label = {intent: idx for idx, intent in enumerate(intent_texts)}
        labels = [intent_to_label[intent_texts[intent]] for intent in intents]
        
        print(f"   Debug - Sample texts: {texts[:3]}")
        print(f"   Debug - Sample labels: {labels[:3]}")
        print(f"   Debug - Label range: {min(labels)} to {max(labels)}")
        
        # Tokenize all texts
        encoded = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='tf'
        )
        
        print(f"   Debug - Encoded input_ids shape: {encoded['input_ids'].shape}")
        print(f"   Debug - Labels array shape: {len(labels)}")
        
        # Create TensorFlow dataset - FIXED: ensure proper tensor creation
        dataset_dict = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
        
        # Handle token_type_ids if present
        if 'token_type_ids' in encoded:
            dataset_dict['token_type_ids'] = encoded['token_type_ids']
        else:
            dataset_dict['token_type_ids'] = tf.zeros_like(encoded['input_ids'])
        
        # FIXED: Convert labels to proper tensor
        labels_tensor = tf.constant(labels, dtype=tf.int64)
        
        tf_dataset = tf.data.Dataset.from_tensor_slices({
            **dataset_dict,
            'labels': labels_tensor
        })
        
        if is_training:
            tf_dataset = tf_dataset.shuffle(1000, seed=42)
        
        tf_dataset = tf_dataset.batch(batch_size)
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
        
        return tf_dataset
    
    train_dataset = prepare_tf_dataset(dataset['train'], train_df, batch_size, is_training=True)
    dev_dataset = prepare_tf_dataset(dataset['validation'], dev_df, batch_size)
    test_dataset = prepare_tf_dataset(dataset['test'], test_df, batch_size)
    
    print("   ✓ TensorFlow datasets created!")
    
    # 5. Setup training - FIXED optimizer setup
    print("\n5️⃣ Đang setup training...")
    
    # Calculate training steps
    train_steps = len(dataset['train']) // batch_size
    total_steps = train_steps * epochs
    warmup_steps = min(1000, total_steps // 10)
    
    print(f"   Debug - Train steps per epoch: {train_steps}")
    print(f"   Debug - Total steps: {total_steps}")
    print(f"   Debug - Warmup steps: {warmup_steps}")
    
    # FIXED: Use simple optimizer instead of create_optimizer (causes issues with TF 2.10)
    # Learning rate và schedule

    # learning_rate = 5e-5

    # lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    #     initial_learning_rate=learning_rate,
    #     decay_steps=total_steps,
    #     end_learning_rate=learning_rate * 0.1
    # )

    # # Dùng Adam optimizer (TF 2.10 không có AdamW)
    # optimizer = tf.keras.optimizers.Adam(
    #     learning_rate=lr_schedule,
    #     epsilon=1e-8
    # )



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
            patience=3,  # Increased patience
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
        # tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor='val_loss',  # Monitor loss instead
        #     factor=0.5,
        #     patience=2,
        #     mode='min',
        #     verbose=1,
        #     min_lr=1e-7
        # )
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
        
        # Print training summary
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        print(f"   📊 Final Training Accuracy: {final_train_acc:.4f}")
        print(f"   📊 Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"   📊 Final Training Loss: {final_train_loss:.4f}")
        print(f"   📊 Final Validation Loss: {final_val_loss:.4f}")
        
    except Exception as e:
        print(f"   ❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 8. Evaluate on test set
    print("\n8️⃣ Đánh giá model...")
    
    try:
        test_results = model.evaluate(test_dataset, verbose=0)
        test_loss = test_results[0]
        test_accuracy = test_results[1]
        
        print(f"   ✓ Test Accuracy: {test_accuracy:.4f}")
        print(f"   ✓ Test Loss: {test_loss:.4f}")
        
    except Exception as e:
        print(f"   ⚠️ Test evaluation error: {e}")
        test_accuracy = final_val_acc  # Fallback to validation accuracy
    
    # 9. Save model
    print("\n9️⃣ Lưu model...")
    
    try:
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
        history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
        with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        # Save training config
        config = {
            'language': language,
            'epochs': epochs,
            'batch_size': batch_size,
            'max_length': MAX_LENGTH,
            'num_labels': num_labels,
            'learning_rate': learning_rate,
            'model_name': 'google/mobilebert-uncased'
        }
        
        with open(os.path.join(OUTPUT_DIR, 'training_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   ✓ Model saved to: {OUTPUT_DIR}")
        print(f"   ✓ Intent mapping saved to: {OUTPUT_DIR}/intent_mapping.json")
        print(f"   ✓ Training config saved to: {OUTPUT_DIR}/training_config.json")
        
    except Exception as e:
        print(f"   ⚠️ Save error: {e}")
    
    # 10. Quick test
    print("\n🔟 Quick test với một số samples...")
    
    test_samples = [
        "set alarm for 7 am tomorrow",
        "play some music", 
        "what's the weather",
        "order pizza",
        "turn off lights"
    ]
    
    try:
        for sample in test_samples:
            inputs = tokenizer(
                sample,
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors='tf'
            )
            
            # Predict
            outputs = model(inputs)
            predictions = tf.nn.softmax(outputs.logits, axis=-1)
            predicted_id = tf.argmax(predictions, axis=-1)[0].numpy()
            confidence = predictions[0][predicted_id].numpy()
            
            # Use intent mapping to get text label
            predicted_intent = intent_mapping.get(predicted_id, f"Unknown_ID_{predicted_id}")
            
            print(f"   📝 '{sample}' -> {predicted_intent} ({confidence:.3f}) | Label ID: {predicted_id}")
    
    except Exception as e:
        print(f"   ⚠️ Quick test error: {e}")
    
    print("\n" + "="*60)
    print("🎉 HOÀN THÀNH!")
    print(f"✅ Model đã được train và lưu tại: {OUTPUT_DIR}")
    print(f"✅ Test accuracy: {test_accuracy:.4f}")
    print("✅ Có thể sử dụng model cho inference!")
    print("="*60)
    
    return True

def quick_inference_demo(model_path=OUTPUT_DIR):
    """Demo nhanh inference - FIXED version"""
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
            label_encoder_path = os.path.join(model_path, 'label_encoder.pkl')
            if os.path.exists(label_encoder_path):
                label_encoder = joblib.load(label_encoder_path)
                intent_mapping = {idx: intent for idx, intent in enumerate(label_encoder.classes_)}
                print("   ⚠️ Using fallback label encoder mapping")
            else:
                print("   ❌ No intent mapping found!")
                return False
        
        print("✅ Model loaded successfully!")
        
        # Load training config if available
        config_path = os.path.join(model_path, 'training_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"   📋 Training config: {config['language']}, {config['epochs']} epochs")
        
        # Show some example intents
        print(f"\n📋 Available intents (showing first 10):")
        for i, intent in list(intent_mapping.items())[:10]:
            print(f"   {i}: {intent}")
        if len(intent_mapping) > 10:
            print(f"   ... và {len(intent_mapping)-10} intents khác")
        
        # Test samples
        test_samples = [
            "set alarm for 7 am",
            "play some music",
            "what's the weather like",
            "turn on the lights",
            "book a restaurant",
            "how much is apple stock",
            "send email to john"
        ]
        
        print(f"\n🧪 Test với các sample:")
        for sample in test_samples:
            inputs = tokenizer(
                sample,
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors='tf'
            )
            
            outputs = model(inputs)
            probabilities = tf.nn.softmax(outputs.logits, axis=-1)
            
            # Top prediction
            predicted_id = tf.argmax(probabilities[0]).numpy()
            confidence = probabilities[0][predicted_id].numpy()
            intent_text = intent_mapping.get(predicted_id, f"Unknown_ID_{predicted_id}")
            
            print(f"   📝 '{sample}' -> {intent_text} ({confidence:.3f})")
        
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
                max_length=MAX_LENGTH,
                return_tensors='tf'
            )
            
            outputs = model(inputs)
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
            batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 16
            
            print(f"🏋️ Training mode - Language: {language}, Epochs: {epochs}, Batch: {batch_size}")
            
            # Check requirements
            if check_and_install_requirements():
                quick_train(language=language, epochs=epochs, batch_size=batch_size)
            
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
                try:
                    choice = input().strip().lower()
                    if choice == 'y':
                        quick_inference_demo()
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()