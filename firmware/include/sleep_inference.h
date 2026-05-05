#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "student_model_data.cc"

enum class SleepStage : uint8_t { W = 0, N1 = 1, N2 = 2, N3 = 3, REM = 4, UNKNOWN = 255 };

class SleepClassifier {
public:
    static constexpr int kFreqBins = 69;
    static constexpr int kTimeSteps = 29;
    static constexpr int kGruHidden = 96;
    static constexpr int kNumStages = 5;
    static constexpr int kTensorArenaBytes = 40 * 1024;

    bool Init() {
        model_ = tflite::GetModel(student_model_data);
        if (model_->version() != TFLITE_SCHEMA_VERSION) {
            return false;
        }

        resolver_.AddConv2D();
        resolver_.AddDepthwiseConv2D();
        resolver_.AddAveragePool2D();
        resolver_.AddFullyConnected();
        resolver_.AddReshape();
        resolver_.AddSoftmax();
        resolver_.AddQuantize();
        resolver_.AddDequantize();

        interpreter_ = std::make_unique<tflite::MicroInterpreter>(
            model_, resolver_, tensor_arena_, sizeof(tensor_arena_));
        if (interpreter_->AllocateTensors() != kTfLiteOk) {
            return false;
        }

        std::fill(gru_state_, gru_state_ + kGruHidden, 0.0f);
        return true;
    }

    SleepStage Classify(const float* spec) {
        if (!interpreter_) {
            return SleepStage::UNKNOWN;
        }

        int8_t* input = interpreter_->input(0)->data.int8;
        const float scale = interpreter_->input(0)->params.scale;
        const int zp = interpreter_->input(0)->params.zero_point;

        for (int i = 0; i < kFreqBins * kTimeSteps; ++i) {
            const float q = spec[i] / scale + static_cast<float>(zp);
            const float clamped = std::max(-128.0f, std::min(127.0f, q));
            input[i] = static_cast<int8_t>(clamped);
        }

        if (interpreter_->Invoke() != kTfLiteOk) {
            return SleepStage::UNKNOWN;
        }

        const int8_t* logits = interpreter_->output(0)->data.int8;
        int predicted = 0;
        for (int i = 1; i < kNumStages; ++i) {
            if (logits[i] > logits[predicted]) {
                predicted = i;
            }
        }

        return static_cast<SleepStage>(predicted);
    }

private:
    uint8_t tensor_arena_[kTensorArenaBytes] = {};
    float gru_state_[kGruHidden] = {};
    tflite::MicroMutableOpResolver<8> resolver_;
    const tflite::Model* model_ = nullptr;
    std::unique_ptr<tflite::MicroInterpreter> interpreter_;
};
