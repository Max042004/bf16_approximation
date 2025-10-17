#include <stdint.h>
typedef struct { uint16_t bits;} bf16_t;
bf16_t fp32_to_bf16(float val);
float bf16_to_fp32 (bf16_t val);
bf16_t bf16_sin(bf16_t a, int *record_k);
