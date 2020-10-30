package com.stupid.styx;

public class ModelConfig {
    public int runtime_ins_ms;
    public int svd_rank;
    public int max_image_size;
    ModelConfig(
            int runtime_ins_ms,
            int svd_rank,
            int max_image_size) {
        this.runtime_ins_ms = runtime_ins_ms;
        this.svd_rank = svd_rank;
        this.max_image_size = max_image_size;
    }
}
