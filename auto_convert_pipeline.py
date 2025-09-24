#!/usr/bin/env python3
# auto_convert_pipeline.py - Fixed version

import torch
import ai_edge_torch
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class AutoConvertPipeline:
    def __init__(self, model_path, output_dir="./output"):
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def convert_and_validate(self):
        print("🚀 Starting auto-conversion pipeline...")
        
        try:
            # 1. Load PyTorch model
            model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # 2. Prepare sample input with explicit types
            sample_text = "play music please"
            sample_input = tokenizer(
                sample_text, 
                return_tensors="pt", 
                max_length=128, 
                truncation=True, 
                padding='max_length'
            )
            
            # 3. Get PyTorch baseline
            model.eval()
            with torch.no_grad():
                pytorch_output = model(**sample_input)
                pytorch_probs = torch.softmax(pytorch_output.logits, dim=-1)
            
            # 4. Create wrapper class to handle input format
            class ModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, input_ids, attention_mask):
                    return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            
            wrapped_model = ModelWrapper(model)
            
            # 5. Prepare inputs with explicit dtype
            input_ids = sample_input['input_ids'].to(torch.int32)
            attention_mask = sample_input['attention_mask'].to(torch.int32)
            
            # 6. Convert to TFLite using the wrapper
            print("🔄 Converting to TFLite...")
            
            # Method 1: Using ai_edge_torch with explicit sample inputs
            tflite_model = ai_edge_torch.convert(
                wrapped_model, 
                (input_ids, attention_mask)
            )
            
            # 7. Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tflite_path = f"{self.output_dir}/mobilebert_intent_{timestamp}.tflite"
            tflite_model.export(tflite_path)
            
            # 8. Validate conversion
            print("🔍 Validating conversion...")
            tflite_output = tflite_model(input_ids, attention_mask)
            tflite_probs = torch.softmax(torch.tensor(tflite_output), dim=-1)
            
            # 9. Compare results
            max_diff = torch.max(torch.abs(pytorch_probs - tflite_probs)).item()
            
            print(f"✅ Conversion successful!")
            print(f"📁 Saved to: {tflite_path}")
            print(f"🎯 Max probability difference: {max_diff:.6f}")
            print(f"📊 PyTorch probs: {pytorch_probs[0].numpy()}")
            print(f"📊 TFLite probs:  {tflite_probs[0].numpy()}")
            
            # 10. Generate metadata
            metadata = {
                "timestamp": timestamp,
                "model_path": self.model_path,
                "tflite_path": tflite_path,
                "max_diff": max_diff,
                "pytorch_prediction": torch.argmax(pytorch_probs).item(),
                "tflite_prediction": torch.argmax(tflite_probs).item(),
                "sample_text": sample_text
            }
            
            return True, metadata
            
        except Exception as e:
            print(f"❌ Conversion failed: {str(e)}")
            print("🔧 Trying alternative conversion methods...")
            
            # Alternative Method: Direct TensorFlow conversion
            try:
                return self._alternative_conversion()
            except Exception as e2:
                print(f"❌ Alternative conversion also failed: {str(e2)}")
                return False, None
    
    def _alternative_conversion(self):
        """Alternative conversion method using TensorFlow directly"""
        print("🔄 Attempting TensorFlow-based conversion...")
        
        import tensorflow as tf
        from transformers import TFAutoModelForSequenceClassification
        
        # Load TensorFlow version of the model
        tf_model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_path, 
            from_tf=False
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Prepare sample input
        sample_text = "play music please"
        sample_input = tokenizer(
            sample_text, 
            return_tensors="tf", 
            max_length=128, 
            truncation=True, 
            padding='max_length'
        )
        
        # Create concrete function
        @tf.function
        def model_func(input_ids, attention_mask):
            return tf_model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        # Get concrete function with proper signatures
        concrete_func = model_func.get_concrete_function(
            input_ids=tf.TensorSpec(shape=[1, 128], dtype=tf.int32),
            attention_mask=tf.TensorSpec(shape=[1, 128], dtype=tf.int32)
        )
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        
        tflite_model = converter.convert()
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tflite_path = f"{self.output_dir}/mobilebert_intent_tf_{timestamp}.tflite"
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ Alternative conversion successful!")
        print(f"📁 Saved to: {tflite_path}")
        
        # Test the converted model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], sample_input['input_ids'].numpy())
        interpreter.set_tensor(input_details[1]['index'], sample_input['attention_mask'].numpy())
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        
        metadata = {
            "timestamp": timestamp,
            "model_path": self.model_path,
            "tflite_path": tflite_path,
            "conversion_method": "tensorflow",
            "sample_text": sample_text
        }
        
        return True, metadata

    def quantize_model(self, tflite_path):
        """Apply post-training quantization to reduce model size"""
        print("🗜️ Applying quantization...")
        
        import tensorflow as tf
        
        # Load the model
        converter = tf.lite.TFLiteConverter.from_saved_model(tflite_path)
        
        # Apply quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self._representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        quantized_model = converter.convert()
        
        # Save quantized model
        quantized_path = tflite_path.replace('.tflite', '_quantized.tflite')
        with open(quantized_path, 'wb') as f:
            f.write(quantized_model)
        
        print(f"✅ Quantized model saved to: {quantized_path}")
        return quantized_path
    
    def _representative_dataset_gen(self):
        """Generate representative dataset for quantization"""
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        sample_texts = [
            "play music please",
            "turn on the lights", 
            "what's the weather",
            "set a timer for 5 minutes",
            "call mom"
        ]
        
        for text in sample_texts:
            sample_input = tokenizer(
                text,
                return_tensors="tf",
                max_length=128,
                truncation=True,
                padding='max_length'
            )
            yield [
                sample_input['input_ids'].numpy().astype('int32'),
                sample_input['attention_mask'].numpy().astype('int32')
            ]

# Enhanced usage script with error handling
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python auto_convert_pipeline.py <model_path>")
        print("Example: python auto_convert_pipeline.py microsoft/DialoGPT-medium")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    print(f"🤖 Converting model: {model_path}")
    pipeline = AutoConvertPipeline(model_path)
    success, metadata = pipeline.convert_and_validate()
    
    if success:
        print(f"\n🎉 Ready for Flutter deployment!")
        model_size_mb = os.path.getsize(metadata['tflite_path']) / 1024 / 1024
        print(f"📏 Model size: {model_size_mb:.2f} MB")
        
        # Optionally apply quantization if model is large
        if model_size_mb > 50:
            print("📦 Model is large, applying quantization...")
            try:
                quantized_path = pipeline.quantize_model(metadata['tflite_path'])
                quantized_size_mb = os.path.getsize(quantized_path) / 1024 / 1024
                print(f"📏 Quantized size: {quantized_size_mb:.2f} MB")
                print(f"💾 Size reduction: {((model_size_mb - quantized_size_mb) / model_size_mb) * 100:.1f}%")
            except Exception as e:
                print(f"⚠️ Quantization failed: {str(e)}")
        
        print(f"\n📋 Conversion Summary:")
        print(f"   Original model: {metadata['model_path']}")
        print(f"   TFLite model: {metadata['tflite_path']}")
        print(f"   Sample text: '{metadata['sample_text']}'")
        if 'max_diff' in metadata:
            print(f"   Accuracy difference: {metadata['max_diff']:.6f}")
    else:
        print("\n❌ Conversion failed. Check the error messages above.")
        print("💡 Common solutions:")
        print("   1. Ensure the model is compatible with TensorFlow Lite")
        print("   2. Try a different model architecture")
        print("   3. Check model permissions and internet connection")
        sys.exit(1)