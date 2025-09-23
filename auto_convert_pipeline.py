#!/usr/bin/env python3
# auto_convert_pipeline.py

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
        
        # 1. Load PyTorch model
        model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # 2. Prepare sample input
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
        
        # 4. Convert to TFLite
        try:
            # Input for conversion (only input_ids and attention_mask)
            convert_input = (
                sample_input['input_ids'], 
                sample_input['attention_mask']
            )
            
            tflite_model = ai_edge_torch.convert(model, convert_input)
            
            # 5. Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tflite_path = f"{self.output_dir}/mobilebert_intent_{timestamp}.tflite"
            tflite_model.export(tflite_path)
            
            # 6. Validate conversion
            tflite_output = tflite_model(*convert_input)
            tflite_probs = torch.softmax(torch.tensor(tflite_output), dim=-1)
            
            # 7. Compare results
            max_diff = torch.max(torch.abs(pytorch_probs - tflite_probs)).item()
            
            print(f"✅ Conversion successful!")
            print(f"📁 Saved to: {tflite_path}")
            print(f"🎯 Max probability difference: {max_diff:.6f}")
            print(f"📊 PyTorch probs: {pytorch_probs[0].numpy()}")
            print(f"📊 TFLite probs:  {tflite_probs[0].numpy()}")
            
            # 8. Generate metadata
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
            return False, None

# Usage script
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python auto_convert_pipeline.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    pipeline = AutoConvertPipeline(model_path)
    success, metadata = pipeline.convert_and_validate()
    
    if success:
        print(f"\n🎉 Ready for Flutter deployment!")
        print(f"Model size: {os.path.getsize(metadata['tflite_path']) / 1024 / 1024:.2f} MB")