#include "sleep_inference.h"

#include <cstdint>

// Placeholder hooks: wire these to your board-specific drivers.
namespace eeg_frontend {
int16_t ReadSample(int channel) {
    (void)channel;
    return 0;
}
void Init(int sample_rate_hz) {
    (void)sample_rate_hz;
}
}

namespace ble_service {
void Init() {}
void NotifyStage(uint8_t stage) {
    (void)stage;
}
}

int main() {
    constexpr int kSampleRate = 250;
    constexpr int kEpochSamples = kSampleRate * 30;

    static int16_t raw_buf[2][kEpochSamples] = {};
    static int write_buf = 0;
    static int sample_n = 0;
    static int buf_ready = -1;

    SleepClassifier classifier;

    ble_service::Init();
    eeg_frontend::Init(kSampleRate);

    if (!classifier.Init()) {
        while (true) {
        }
    }

    float spectrogram[SleepClassifier::kFreqBins * SleepClassifier::kTimeSteps] = {};

    while (true) {
        const int16_t sample = eeg_frontend::ReadSample(0);
        raw_buf[write_buf][sample_n++] = sample;

        if (sample_n >= kEpochSamples) {
            buf_ready = write_buf;
            write_buf ^= 1;
            sample_n = 0;
        }

        if (buf_ready >= 0) {
            // TODO: Replace with real filter + STFT pipeline.
            for (float& v : spectrogram) {
                v = 0.0f;
            }

            const SleepStage stage = classifier.Classify(spectrogram);
            ble_service::NotifyStage(static_cast<uint8_t>(stage));
            buf_ready = -1;
        }
    }
}
