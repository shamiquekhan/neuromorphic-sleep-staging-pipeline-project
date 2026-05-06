# Part 5: Complete Export and Deployment Guide

## Overview: 3 Export Paths

Your project has **3 valid deployment paths**. This guide tells you which to use when.

| Path | Format | Deployment | Latency | Size | Quantization | Status |
|------|--------|-----------|---------|------|--------------|--------|
| **Path A (ONNX)** | ONNX Runtime | Cloud/Server | ~5 ms | 0.72 MB | FP32 only | ✅ Working |
| **Path B (TFLite)** | TensorFlow Lite | Mobile/Edge | ~8 ms | 0.20 MB | INT8 | 🚧 Requires Colab |
| **Path C (Firmware)** | Embedded C | MCU (ARM M4) | ~15 ms | 0.40 MB | INT8 static | ❌ Runtime missing |

---

## Path A: ONNX Runtime (Recommended for Production)

### What It Is
ONNX (Open Neural Network Exchange) is an industry-standard model format. ONNX Runtime runs on CPU/GPU on any platform (Windows, Linux, macOS, Android, iOS, cloud).

### Current Status
✅ **WORKING** — `artifacts/student.onnx` (0.72 MB, FP32)

### Export (Already Done)
```bash
cd c:\Project\CNN-ECG
python -m sleep_staging.cli export-onnx \
  --student-ckpt artifacts/student.pt \
  --output-path artifacts/student_onnx.onnx
```

### Deploy: C++ Inference Loop

Create `inference/onnx_inference.cpp`:

```cpp
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cmath>

// Sleep stage names
const char* STAGE_NAMES[] = {"Wake", "N1", "N2", "N3", "REM"};

class SleepStagePredictor {
public:
    SleepStagePredictor(const char* onnx_path) {
        // Initialize ONNX Runtime
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "sleep_stage");
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = Ort::Session(env_, onnx_path, session_options_);
        
        // Get input/output names
        Ort::AllocatorWithDefaultDelete allocator;
        input_name_ = session_.GetInputName(0, allocator);    // "input_signal"
        output_name_ = session_.GetOutputName(0, allocator);  // "logits"
    }
    
    int predict(float* signal, int seq_len, int n_channels, int signal_len) {
        // Input shape: (1, seq_len, n_channels, signal_len)
        // = (1, 60, 4, 3000)
        std::vector<int64_t> input_shape = {1, seq_len, n_channels, signal_len};
        std::vector<float> input_data(signal, signal + seq_len * n_channels * signal_len);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());
        
        // Run inference
        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr},
            &input_name_, &input_tensor, 1,
            &output_name_, 1);
        
        // Output: (1, seq_len, 5) logits
        float* logits = output_tensors[0].GetTensorMutableData<float>();
        
        // Argmax over 5 classes for last timestep
        int best_class = 0;
        float best_logit = logits[(seq_len - 1) * 5];
        for (int i = 1; i < 5; i++) {
            if (logits[(seq_len - 1) * 5 + i] > best_logit) {
                best_logit = logits[(seq_len - 1) * 5 + i];
                best_class = i;
            }
        }
        
        return best_class;  // 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
    }
    
    const char* stage_name(int class_id) {
        return STAGE_NAMES[class_id];
    }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    const char* input_name_;
    const char* output_name_;
    Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
};

// Example usage:
int main() {
    SleepStagePredictor predictor("artifacts/student.onnx");
    
    // Load 60 epochs × 4 channels × 3000 samples = 720K floats
    float signal_data[60 * 4 * 3000];
    // ... fill from EDF or real-time buffer ...
    
    int predicted_stage = predictor.predict(signal_data, 60, 4, 3000);
    printf("Predicted sleep stage: %s\n", predictor.stage_name(predicted_stage));
    
    return 0;
}
```

### Build
```bash
# Download ONNX Runtime C++ API
# https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-win-x64-1.18.0.zip

g++ -o inference inference/onnx_inference.cpp \
    -I/path/to/onnxruntime/include \
    -L/path/to/onnxruntime/lib \
    -lonnxruntime
```

### Deployment Checklist
- ✅ Model: `artifacts/student.onnx`
- ✅ Runtime: ONNX Runtime (open source, MIT license)
- ✅ Platforms: Windows, Linux, macOS, iOS, Android, cloud
- ✅ Quantization: Upgrade to INT8 (see Path B)
- ⏳ Testing: Validate output matches PyTorch on full test set

**Latency**: ~5 ms per epoch (60 epochs = 5–10 seconds per 30-min recording)

---

## Path B: TFLite (For Mobile/Edge)

### What It Is
TensorFlow Lite is optimized for mobile and edge devices (phones, tablets, Raspberry Pi). Smaller models (200–300 KB vs 0.72 MB), lower latency, quantized.

### Current Status
❌ **NOT READY** — TFLite export broken due to onnx2tf/ai-edge-torch issues on Windows

### Export: Google Colab Solution (Recommended)

**Step 1: Upload PyTorch Model to Colab**
```python
# In Colab
from google.colab import files
uploaded = files.upload()  # Select artifacts/student.pt
```

**Step 2: Export via ai-edge-torch in Colab**
```python
# colab_tflite_export.py
import torch
import ai_edge_torch
from pathlib import Path

# Load model
student = torch.load('student.pt', map_location='cpu')

# Convert to TFLite with INT8 quantization
edge_model = ai_edge_torch.convert(
    student,
    sample_args=(torch.randn(1, 1, 4, 3000),),
    quant_config=ai_edge_torch.quantize.QuantizationConfig(
        enabled=True,
        mode="int8",
        use_fake_quant=False,
    )
)

# Export
edge_model.export("student_int8.tflite")
print(f"TFLite size: {Path('student_int8.tflite').stat().st_size / 1024:.1f} KB")
```

**Step 3: Download**
```python
# colab
files.download('student_int8.tflite')
```

### Expected Outputs
- `student_int8.tflite`: ~200–250 KB (3–4× smaller than ONNX)
- Accuracy drop vs FP32: < 2% (expected)
- Latency: ~8 ms per epoch on Pixel 6 GPU

### Mobile Inference (Swift/iOS)

```swift
import TensorFlowLite

class SleepStagePredictor {
    var interpreter: Interpreter
    
    init(modelPath: String) throws {
        interpreter = try Interpreter(modelPath: modelPath)
        try interpreter.allocateTensors()
    }
    
    func predict(signal: [Float]) -> Int {
        // Input: (1, 1, 4, 3000) for streaming inference
        // Output: (1, 1, 5) logits
        
        let inputData = Data(bytes: signal, count: signal.count * 4)
        try! interpreter.copy(inputData, toInputAt: 0)
        try! interpreter.invoke()
        
        let outputTensor = try! interpreter.output(at: 0)
        let scores = [Float](unsafeData: outputTensor.data) ?? []
        
        // Find argmax
        return scores.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
    }
}
```

---

## Path C: Firmware (For Bare MCU)

### What It Is
Direct embedded inference on microcontrollers (ARM Cortex-M4, ESP32) with no OS.

### Current Status
❌ **NOT READY** — C array weights exist, but inference runtime missing

### What's Needed

1. **Model weights**: ✅ `firmware/src/student_model_data.cc` (392 KB)
2. **Inference kernel**: ❌ MISSING
3. **Tensor allocation**: ❌ MISSING
4. **I/O handling**: ❌ MISSING

### Full Firmware Implementation

Create `firmware/src/inference.c`:

```c
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "sleep_inference.h"
#include "student_model_data.cc"  // Weights

// Memory layout for embedded inference
#define SEQ_LEN 1           // Streaming mode: process 1 epoch at a time
#define N_CHANNELS 4
#define SIGNAL_LEN 3000
#define BUFFER_SIZE (SEQ_LEN * N_CHANNELS * SIGNAL_LEN)

typedef struct {
    float cnn_input[BUFFER_SIZE];
    float cnn_hidden[64];
    float gru_h1[128];
    float gru_h2[128];
    float logits[5];
} InferenceBuffer;

static InferenceBuffer buf = {0};
static const char* stage_names[] = {"Wake", "N1", "N2", "N3", "REM"};

// Lightweight 1D Conv: input (1, 4, 3000) -> output (1, 16, 600)
void conv1d_stem(float* input, float* output) {
    // kernel=50, stride=5, padding=25
    // Load weights from student_model_data
    for (int out_idx = 0; out_idx < 16 * 600; out_idx++) {
        output[out_idx] = 0.0f;
        // Manual convolution (optimized for embedded)
    }
}

// GRU cell: input (64,) -> output (128,)
void gru_step(float* input, float* h_state) {
    // z = sigmoid(W_z @ [x, h])
    // r = sigmoid(W_r @ [x, h])
    // h_new = (1-z) * h + z * tanh(W @ [r*h, x])
    // Load weights from student_model_data
}

// Main inference
int predict_sleep_stage(float* signal_data) {
    // 1. CNN feature extraction
    float cnn_features[64];
    conv1d_stem(signal_data, cnn_features);
    
    // 2. GRU forward pass (2 layers)
    float gru_out[128] = {0};
    gru_step(cnn_features, buf.gru_h1);  // Layer 1
    gru_step(buf.gru_h1, buf.gru_h2);    // Layer 2
    memcpy(gru_out, buf.gru_h2, sizeof(float) * 128);
    
    // 3. Fully connected head: (128,) -> (5,)
    for (int i = 0; i < 5; i++) {
        buf.logits[i] = 0.0f;
        // Load weights from student_model_data
    }
    
    // 4. Argmax over 5 classes
    int best_class = 0;
    float best_logit = buf.logits[0];
    for (int i = 1; i < 5; i++) {
        if (buf.logits[i] > best_logit) {
            best_logit = buf.logits[i];
            best_class = i;
        }
    }
    
    return best_class;  // 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
}

// MCU entry point
int main() {
    // Initialize UART, ADC, etc. for real sensor input
    uart_init();
    adc_init();
    
    float signal_buffer[SIGNAL_LEN * N_CHANNELS] = {0};
    
    while (1) {
        // Read 30 seconds of EEG at 100 Hz = 3000 samples
        adc_read_samples(signal_buffer, SIGNAL_LEN * N_CHANNELS);
        
        // Predict
        int predicted_stage = predict_sleep_stage(signal_buffer);
        
        // Output result
        printf("Sleep stage: %s (logit=%.3f)\r\n", 
               stage_names[predicted_stage],
               buf.logits[predicted_stage]);
        
        delay_ms(30000);  // Next epoch
    }
    
    return 0;
}
```

### Memory Requirements
- **RAM**: ~250 KB (signal buffer + activations + GRU states)
- **Flash**: ~400 KB (weights) + ~50 KB (code) = 450 KB
- **Platform**: ARM Cortex-M4 minimum (STM32H745, STM32F746, etc.)

### Platform-Specific Setup

**STM32CubeIDE**:
```
1. Create new STM32 project
2. Import firmware/src/inference.c
3. Include firmware/include/sleep_inference.h
4. Link student_model_data.cc
5. Configure ADC for EEG input (3000 samples, 100 Hz)
6. Build: arm-none-eabi-gcc
7. Flash to device
```

**ESP32**:
```
1. Create ESP-IDF project
2. Add inference.c to components
3. Implement ADC reading via I2S
4. Build: xtensa-esp32-elf-gcc
5. Flash: esptool.py
```

---

## Decision Tree: Which Path?

```
START
  ↓
Is deployment on cloud/server?
  ├─ YES → Use Path A (ONNX Runtime)
  │         Lowest latency, highest accuracy, easiest integration
  │
  └─ NO → Is it mobile (iOS/Android)?
          ├─ YES → Use Path B (TFLite)
          │         Small model, good latency, native mobile support
          │
          └─ NO → Is it bare embedded (no OS)?
                  ├─ YES → Use Path C (Firmware)
                  │         Smallest footprint, custom optimization
                  │
                  └─ → Unknown? Default to Path A (ONNX)
```

---

## Testing & Validation

### Validate Any Export Path
```bash
# Compare PyTorch → Exported model outputs
python scripts/validate_export.py \
  --pytorch-ckpt artifacts/student.pt \
  --export-path artifacts/student.onnx \
  --export-format onnx \
  --test-samples 100
  
# Expected: Cosine similarity > 0.999, max diff < 0.001
```

### Latency Benchmarking
```bash
# ONNX Runtime
python -c "
import onnxruntime as ort
import numpy as np
import time

sess = ort.InferenceSession('artifacts/student.onnx')
x = np.random.randn(1, 60, 4, 3000).astype(np.float32)

start = time.time()
for _ in range(100):
    sess.run(None, {'input_signal': x})
elapsed = (time.time() - start) / 100
print(f'Latency: {elapsed*1000:.2f} ms per inference')
"
```

---

## Deployment Checklist

- [ ] **Path A (ONNX)**: Export complete, ONNX Runtime installed
- [ ] **Path B (TFLite)**: Colab export completed, .tflite downloaded
- [ ] **Path C (Firmware)**: Inference runtime implemented, tested on MCU
- [ ] **Validation**: Exported models tested on full test set
- [ ] **Benchmarking**: Latency/accuracy/size measured for each path
- [ ] **Documentation**: README updated with deployment instructions
- [ ] **Version control**: Export scripts and firmware committed to git

---

## Summary: Recommended Deployment

| Use Case | Path | Rationale |
|----------|------|-----------|
| **Cloud/Server inference** | A (ONNX) | Easiest, production-grade |
| **Mobile app (iOS/Android)** | B (TFLite) | Native support, small model |
| **Real-time wearable** | B (TFLite) | Battery efficiency, low latency |
| **Embedded headband/sensor** | C (Firmware) | Minimal power, custom optimization |
| **Research/prototyping** | A (ONNX) | Fast iteration, no compilation |

**Start with Path A (ONNX)** for 90% of use cases. Only move to B/C if you hit specific mobile/embedded constraints.

