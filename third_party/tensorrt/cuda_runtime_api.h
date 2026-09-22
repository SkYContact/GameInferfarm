// cuda_runtime_api.h - minimal type stub (2026-09-20 TRT integration)
// TRT headers consume ONLY cudaStream_t / cudaEvent_t (grep-verified over
// NvInfer*.h). Declaring them here avoids adding a CUDA /I to the ai_duel
// project. Forward-declared opaque struct pointers match the real CUDA ABI.
#ifndef YGO_TRT_CUDA_TYPES_STUB
#define YGO_TRT_CUDA_TYPES_STUB
#ifdef __cplusplus
extern "C" {
#endif
struct CUstream_st;
struct CUevent_st;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
#ifdef __cplusplus
}
#endif
#endif
