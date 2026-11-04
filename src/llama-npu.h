#pragma once

#include "llama-memory.h"

struct llama_model;
struct llama_batch_ext;

// Model-only execution. This is not a ggml graph backend.
class llama_npu : public llama_memory_i {
public:
    uint32_t vocabulary_size() const;
    uint32_t context_size() const;
    uint32_t layer_count() const;

    llama_npu(const llama_model & model, const llama_context_params & params);
    ~llama_npu() override;

    void evaluate(const llama_batch_ext & batch, uint32_t n_ctx, float * logits);
    size_t position() const;
    bool usable() const;

    llama_memory_context_ptr init_batch(llama_batch_allocr &, uint32_t, bool) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context *, bool) override;
    bool get_can_shift() const override { return false; }
    void clear(bool data) override;
    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    void seq_cp(llama_seq_id, llama_seq_id, llama_pos, llama_pos) override;
    void seq_keep(llama_seq_id) override;
    void seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) override;
    void seq_div(llama_seq_id, llama_pos, llama_pos, int) override;
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override { return {}; }
    void state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const override;
    void state_read(llama_io_read_i &, llama_seq_id, llama_state_seq_flags) override;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
    // Void memory APIs cannot return errors. Latch rejection and make decode fail.
    bool rejected = false;
    void reject(const char * operation);
};