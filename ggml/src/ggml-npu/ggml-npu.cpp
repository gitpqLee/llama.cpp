#include "ggml-npu.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <exception>
#include <new>

#ifdef GGML_NPU_RUNTIME
#include <npu_runtime.hpp>
#endif

static bool ggml_npu_detect(void) {
#ifdef GGML_NPU_RUNTIME
    try {
        // The runtime selects one NPU and does not expose device enumeration.
        npu::Device device;
        // This backend currently ships the 5010 / six-worker model ABI only.
        return device.getDeviceKey() == "5010" && device.getTileCount() == 3;
    } catch (const std::exception & e) {
        GGML_LOG_DEBUG("%s: NPU unavailable: %s\n", __func__, e.what());
    } catch (...) {
        GGML_LOG_DEBUG("%s: NPU runtime initialization failed\n", __func__);
    }
#endif
    return false;
}

static const char * ggml_backend_npu_name(ggml_backend_t) {
    return "NPU";
}

static void ggml_backend_npu_free(ggml_backend_t backend) {
    delete backend;
}

static enum ggml_status ggml_backend_npu_graph_compute(ggml_backend_t, ggml_cgraph * graph) {
    if (!graph || ggml_graph_n_nodes(graph) != 0) {
        GGML_LOG_ERROR("NPU: GGML graph execution is unsupported; use the whole-model DSL session\n");
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_npu_i = {
    /* .get_name                = */ ggml_backend_npu_name,
    /* .free                    = */ ggml_backend_npu_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_npu_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t ggml_backend_npu_guid(void) {
    static ggml_guid guid = { 0x92, 0x0d, 0xc2, 0xf0, 0xc6, 0x82, 0x4c, 0x12,
                              0x91, 0xbc, 0x6a, 0x63, 0x93, 0xe7, 0x2a, 0x15 };
    return &guid;
}

bool ggml_backend_is_npu(ggml_backend_t backend) {
    return backend && ggml_guid_matches(backend->guid, ggml_backend_npu_guid());
}

static const char * ggml_backend_npu_device_get_name(ggml_backend_dev_t) {
    return "NPU";
}

static const char * ggml_backend_npu_device_get_description(ggml_backend_dev_t) {
    return "Intel NPU complete DSL model";
}

static void ggml_backend_npu_device_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_npu_device_get_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_npu_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    *props = {};
    props->name        = ggml_backend_npu_device_get_name(dev);
    props->description = ggml_backend_npu_device_get_description(dev);
    props->type        = ggml_backend_npu_device_get_type(dev);
}

static ggml_backend_t ggml_backend_npu_device_init(ggml_backend_dev_t dev, const char * params) {
    if (params && params[0] != '\0') {
        return nullptr;
    }
    return new (std::nothrow) ggml_backend {
        /* .guid    = */ ggml_backend_npu_guid(),
        /* .iface   = */ ggml_backend_npu_i,
        /* .device  = */ dev,
        /* .context = */ nullptr,
    };
}

static ggml_backend_buffer_type_t ggml_backend_npu_device_get_buffer_type(ggml_backend_dev_t) {
    // Host staging only; the DSL session owns all model and device memory.
    return ggml_backend_cpu_buffer_type();
}

static bool ggml_backend_npu_device_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    return false;
}

static bool ggml_backend_npu_device_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_cpu_buffer_type();
}

static const ggml_backend_device_i ggml_backend_npu_device_i = {
    /* .get_name             = */ ggml_backend_npu_device_get_name,
    /* .get_description      = */ ggml_backend_npu_device_get_description,
    /* .get_memory           = */ ggml_backend_npu_device_get_memory,
    /* .get_type             = */ ggml_backend_npu_device_get_type,
    /* .get_props            = */ ggml_backend_npu_device_get_props,
    /* .init_backend         = */ ggml_backend_npu_device_init,
    /* .get_buffer_type      = */ ggml_backend_npu_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_npu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_npu_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static const char * ggml_backend_npu_reg_get_name(ggml_backend_reg_t) {
    return "NPU";
}

static size_t ggml_backend_npu_reg_get_device_count(ggml_backend_reg_t) {
    static const bool available = ggml_npu_detect();
    return available ? 1 : 0;
}

static ggml_backend_dev_t ggml_backend_npu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    if (index >= ggml_backend_npu_reg_get_device_count(reg)) {
        return nullptr;
    }
    static ggml_backend_device device = {
        /* .iface   = */ ggml_backend_npu_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &device;
}

ggml_backend_reg_t ggml_backend_npu_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ {
            /* .get_name         = */ ggml_backend_npu_reg_get_name,
            /* .get_device_count = */ ggml_backend_npu_reg_get_device_count,
            /* .get_device       = */ ggml_backend_npu_reg_get_device,
            /* .get_proc_address = */ nullptr,
        },
        /* .context = */ nullptr,
    };
    return &reg;
}

ggml_backend_t ggml_backend_npu_init(void) {
    ggml_backend_dev_t dev = ggml_backend_npu_reg_get_device(ggml_backend_npu_reg(), 0);
    return dev ? ggml_backend_npu_device_init(dev, nullptr) : nullptr;
}

GGML_BACKEND_DL_IMPL(ggml_backend_npu_reg)