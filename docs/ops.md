# GGML Operations

List of GGML operations and backend support status.

The local RocketKV CPU/Metal stable top-k supports up to 32768 entries per row, with ascending original indices on equal scores. Metal uses 256-entry blocks and GPU merges for rows above 512 entries selecting at most 256 indices; other shapes retain the single-block or 8192-entry-block paths. Its paired K/V gather preserves F16/F32 values without an intermediate type conversion.

For nonnegative, page-constant token probabilities, RocketKV can select pages and expand them back to the exact token top-k, accounting for the partial live page. CPU and Metal test this path against full token sorting, including zero scores and ties.

## How to add a backend to this table:

1. Run `test-backend-ops support --output csv` with your backend name and redirect output to a csv file in `docs/ops/` (e.g., `docs/ops/CUDA.csv`)
2. Regenerate `/docs/ops.md` via `./scripts/create_ops_docs.py`

Legend:
- ✅ Fully supported by this backend
- 🟡 Partially supported by this backend
- ❌ Not supported by this backend

| Operation | BLAS | CANN | CPU | CUDA | ET | HTP | MTL | OpenCL | SYCL | Vulkan | WebGPU | ZenDNN | zDNN |
|-----------|------|------|------|------|------|------|------|------|------|------|------|------|------|
|                              ABS | ❌ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              ACC | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              ADD | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             ADD1 | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
|                           ADD_ID | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                           ARANGE | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                           ARGMAX | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          ARGSORT | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             CEIL | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                            CLAMP | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        COL2IM_1D | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                           CONCAT | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                             CONT | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                          CONV_2D | ❌ | ❌ | 🟡 | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ❌ | ❌ |
|                       CONV_2D_DW | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          CONV_3D | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                CONV_TRANSPOSE_1D | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                CONV_TRANSPOSE_2D | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              COS | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                      COUNT_EQUAL | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              CPY | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ |
|               CROSS_ENTROPY_LOSS | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|          CROSS_ENTROPY_LOSS_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                           CUMSUM | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             DIAG | ❌ | ❌ | ✅ | ✅ | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                    DIAG_MASK_INF | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              DIV | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                     DSV4_HC_COMB | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                     DSV4_HC_POST | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                      DSV4_HC_PRE | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                              DUP | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              ELU | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              EXP | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                            EXPM1 | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             FILL | ❌ | ❌ | ✅ | ✅ | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                   FLASH_ATTN_EXT | ❌ | 🟡 | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ |
|                            FLOOR | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                  GATED_DELTA_NET | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | 🟡 | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                GATED_LINEAR_ATTN | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            GEGLU | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        GEGLU_ERF | ❌ | ✅ | ✅ | ✅ | 🟡 | ❌ | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                      GEGLU_QUICK | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             GELU | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         GELU_ERF | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                       GELU_QUICK | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         GET_ROWS | ❌ | 🟡 | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ❌ | ❌ |
|                    GET_ROWS_BACK | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | ❌ |
|                       GROUP_NORM | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                      HARDSIGMOID | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        HARDSWISH | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                           IM2COL | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        IM2COL_3D | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                          L2_NORM | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                       LEAKY_RELU | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | 🟡 | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                LIGHTNING_INDEXER | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                              LOG | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             MEAN | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 🟡 | ✅ | ❌ | ❌ | ❌ |
|                              MUL | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          MUL_MAT | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
|                 MUL_MAT_HADAMARD | ❌ | ❌ | ✅ | ❌ | ✅ | 🟡 | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                       MUL_MAT_ID | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 🟡 | ❌ |
|                  MUL_MAT_ID_W4A4 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
|                  MUL_MAT_ID_W4A8 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
|                     MUL_MAT_W4A4 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
|                     MUL_MAT_W4A8 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
|                              NEG | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             NORM | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                   OPT_STEP_ADAMW | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                     OPT_STEP_SGD | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                         OUT_PROD | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | 🟡 |
|                              PAD | ❌ | 🟡 | ✅ | 🟡 | ❌ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ |
|                   PAD_REFLECT_1D | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                          POOL_1D | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                          POOL_2D | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            REGLU | ❌ | ✅ | ✅ | ✅ | 🟡 | ❌ | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             RELU | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                           REPEAT | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                      REPEAT_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                         RMS_NORM | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                    RMS_NORM_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                         ROCKETKV | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
|                             ROLL | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                             ROPE | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        ROPE_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            ROUND | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        RWKV_WKV6 | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                        RWKV_WKV7 | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            SCALE | ❌ | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              SET | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ |
|                         SET_ROWS | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ |
|                              SGN | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          SIGMOID | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             SILU | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        SILU_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
|                              SIN | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         SOFTPLUS | ❌ | ❌ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         SOFT_MAX | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                    SOFT_MAX_BACK | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | ✅ | ❌ | ❌ | ❌ |
|                        SOLVE_TRI | ❌ | ❌ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ |
|                              SQR | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             SQRT | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         SSM_CONV | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         SSM_SCAN | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | 🟡 | 🟡 | ✅ | ❌ | ❌ |
|                             STEP | ❌ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              SUB | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              SUM | ❌ | 🟡 | ✅ | 🟡 | ❌ | 🟡 | 🟡 | ❌ | 🟡 | 🟡 | 🟡 | ❌ | ❌ |
|                         SUM_ROWS | ❌ | ✅ | ✅ | 🟡 | ❌ | 🟡 | ✅ | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ |
|                           SWIGLU | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                     SWIGLU_CLAMP | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                       SWIGLU_OAI | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             TANH | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|               TIMESTEP_EMBEDDING | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            TOP_K | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | 🟡 | ✅ | ❌ | ❌ |
|                              TRI | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                            TRUNC | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          UPSCALE | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                            XIELU | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
