#pragma once

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace npu {

class Device {
public:
    Device() {
        const char * mode = std::getenv("GGML_NPU_MOCK_MODE");
        if (mode && std::strcmp(mode, "exception") == 0) {
            throw std::runtime_error("mock: NPU driver unavailable");
        }
        if (mode && std::strcmp(mode, "unknown") == 0) {
            throw 1;
        }
        if (!mode || std::strcmp(mode, "empty") != 0) {
            key = mode && std::strcmp(mode, "other-device") == 0 ? "4000" : "5010";
        }
        tiles = mode && std::strcmp(mode, "other-topology") == 0 ? 6 : 3;
    }

    const std::string & getDeviceKey() const { return key; }
    std::size_t getTileCount() const { return tiles; }

private:
    std::string key;
    std::size_t tiles = 0;
};

} // namespace npu