// Fused cross-cell token-mixing MLP (TRMMixer.token_mix) — forward + backward CUDA kernels.
//
// Replaces `inp + Linear(TH,NC)(GELU(Linear(NC,TH)(inp.transpose)))`.transpose() with two kernels
// that never leave the natural (B, N_CELLS, in_dim) layout (no transpose/copy), tile over in_dim,
// and cache the (tiny) token-mix weights in shared memory once per block.
//
// Register-budget note (why the backward looks the way it does): an earlier version recomputed
// `pre` in the backward and kept a length-TOKEN_HIDDEN scratch array live alongside two
// length-N_CELLS arrays; ptxas ran out of registers at N_CELLS=36/TOKEN_HIDDEN=64 and spilled
// ~8.4KB/thread to local memory (~1GB of spill traffic per call — the actual bottleneck behind
// several earlier layout/reduction attempts that all measured a regression they didn't explain).
// Fix: never materialize the length-TOKEN_HIDDEN vector — stream over j and accumulate directly
// into a length-N_CELLS output vector, and save `pre` from the forward instead of recomputing it
// (trades ~13MB of extra write+read for ~1GB of spill traffic).
//
// CUDA-graph-capture note (load-bearing): kernels launch on `at::cuda::getCurrentCUDAStream()`,
// not the implicit legacy default stream. `torch.cuda.graph()` capture redirects "current stream"
// to a dedicated capture stream; a kernel launched via bare `<<<grid,block>>>` (stream 0) races
// against the rest of the captured graph instead of being ordered against it — invisible in eager
// use (current stream usually IS the default stream there), but produces silently-wrong gradients
// under replay (parameters visibly update, loss does not converge). Always launch on
// `getCurrentCUDAStream()` in anything meant to be graph-capturable.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <int N_CELLS, int TOKEN_HIDDEN, int C_TILE>
__global__ void tm_fwd(
    const float* __restrict__ inp,    // (B, N_CELLS, in_dim)
    const float* __restrict__ w1,     // (TOKEN_HIDDEN, N_CELLS)
    const float* __restrict__ b1,     // (TOKEN_HIDDEN,)
    const float* __restrict__ w2,     // (N_CELLS, TOKEN_HIDDEN)
    const float* __restrict__ b2,     // (N_CELLS,)
    float* __restrict__ out,          // (B, N_CELLS, in_dim)
    float* __restrict__ h_out,        // (B, TOKEN_HIDDEN, in_dim)
    float* __restrict__ pre_out,      // (B, TOKEN_HIDDEN, in_dim)
    int in_dim
) {
    __shared__ float sw1[TOKEN_HIDDEN * N_CELLS];
    __shared__ float sw2[N_CELLS * TOKEN_HIDDEN];
    __shared__ float sb1[TOKEN_HIDDEN];
    __shared__ float sb2[N_CELLS];
    for (int k = threadIdx.x; k < TOKEN_HIDDEN * N_CELLS; k += blockDim.x) sw1[k] = w1[k];
    for (int k = threadIdx.x; k < N_CELLS * TOKEN_HIDDEN; k += blockDim.x) sw2[k] = w2[k];
    for (int k = threadIdx.x; k < TOKEN_HIDDEN; k += blockDim.x) sb1[k] = b1[k];
    for (int k = threadIdx.x; k < N_CELLS; k += blockDim.x) sb2[k] = b2[k];
    __syncthreads();

    int b = blockIdx.y;
    int c = blockIdx.x * C_TILE + threadIdx.x;
    if (c >= in_dim) return;

    float v[N_CELLS];
    float oi[N_CELLS];                      // accumulator - replaces the live h[TOKEN_HIDDEN]

    const float* base = inp + (size_t)b * N_CELLS * in_dim + c;
#pragma unroll
    for (int i = 0; i < N_CELLS; i++) v[i] = base[(size_t)i * in_dim];
#pragma unroll
    for (int i = 0; i < N_CELLS; i++) oi[i] = v[i] + sb2[i];   // residual + readout bias

    float* hbase = h_out + (size_t)b * TOKEN_HIDDEN * in_dim + c;
    float* pbase = pre_out + (size_t)b * TOKEN_HIDDEN * in_dim + c;
#pragma unroll 4
    for (int j = 0; j < TOKEN_HIDDEN; j++) {
        float pre = sb1[j];
        const float* w1row = sw1 + j * N_CELLS;
#pragma unroll
        for (int i = 0; i < N_CELLS; i++) pre += v[i] * w1row[i];
        float g = 0.5f * pre * (1.0f + erff(pre * 0.7071067811865476f));
        pbase[(size_t)j * in_dim] = pre;     // coalesced
        hbase[(size_t)j * in_dim] = g;       // coalesced
#pragma unroll
        for (int i = 0; i < N_CELLS; i++) oi[i] += g * sw2[i * TOKEN_HIDDEN + j];
    }

    float* obase = out + (size_t)b * N_CELLS * in_dim + c;
#pragma unroll
    for (int i = 0; i < N_CELLS; i++) obase[(size_t)i * in_dim] = oi[i];
}

template <int N_CELLS, int TOKEN_HIDDEN, int C_TILE>
__global__ void tm_bwd(
    const float* __restrict__ w1,        // (TOKEN_HIDDEN, N_CELLS)
    const float* __restrict__ w2,        // (N_CELLS, TOKEN_HIDDEN)
    const float* __restrict__ pre_h,     // (B, TOKEN_HIDDEN, in_dim)  saved by forward
    const float* __restrict__ grad_out,  // (B, N_CELLS, in_dim)
    float* __restrict__ grad_inp,        // (B, N_CELLS, in_dim)
    float* __restrict__ dpre_h,          // (B, TOKEN_HIDDEN, in_dim)
    int in_dim
) {
    __shared__ float sw1[TOKEN_HIDDEN * N_CELLS];
    __shared__ float sw2[N_CELLS * TOKEN_HIDDEN];
    for (int k = threadIdx.x; k < TOKEN_HIDDEN * N_CELLS; k += blockDim.x) sw1[k] = w1[k];
    for (int k = threadIdx.x; k < N_CELLS * TOKEN_HIDDEN; k += blockDim.x) sw2[k] = w2[k];
    __syncthreads();

    int b = blockIdx.y;
    int c = blockIdx.x * C_TILE + threadIdx.x;
    if (c >= in_dim) return;

    float goc[N_CELLS];
    float gi[N_CELLS];                      // accumulator - replaces the live dph[TOKEN_HIDDEN]

    const float* gbase = grad_out + (size_t)b * N_CELLS * in_dim + c;
#pragma unroll
    for (int i = 0; i < N_CELLS; i++) goc[i] = gbase[(size_t)i * in_dim];
#pragma unroll
    for (int i = 0; i < N_CELLS; i++) gi[i] = goc[i];          // residual path

    const float* prebase = pre_h + (size_t)b * TOKEN_HIDDEN * in_dim + c;
    float* dphbase = dpre_h + (size_t)b * TOKEN_HIDDEN * in_dim + c;
#pragma unroll 4
    for (int j = 0; j < TOKEN_HIDDEN; j++) {
        float pre = prebase[(size_t)j * in_dim];               // coalesced read, no recompute
        float up = 0.0f;
#pragma unroll
        for (int i = 0; i < N_CELLS; i++) up += sw2[i * TOKEN_HIDDEN + j] * goc[i];
        float Phi = 0.5f * (1.0f + erff(pre * 0.7071067811865476f));
        float phi = 0.3989422804014327f * expf(-0.5f * pre * pre);
        float d = up * (Phi + pre * phi);                      // gelu'(x) = Phi(x) + x*phi(x)
        dphbase[(size_t)j * in_dim] = d;                       // coalesced
        const float* w1row = sw1 + j * N_CELLS;
#pragma unroll
        for (int i = 0; i < N_CELLS; i++) gi[i] += w1row[i] * d;
    }

    float* dvbase = grad_inp + (size_t)b * N_CELLS * in_dim + c;
#pragma unroll
    for (int i = 0; i < N_CELLS; i++) dvbase[(size_t)i * in_dim] = gi[i];
}

constexpr int C_TILE = 128;

// (N_CELLS, TOKEN_HIDDEN) pairs compiled below. N_CELLS/TOKEN_HIDDEN are compile-time template
// params (needed for register-resident unrolling), so this is a closed set, not an arbitrary
// shape — supporting a new (n_cells, token_hidden) combo means adding an FWD/BWD entry here and
// to models/mixer.py's SUPPORTED_SHAPES, then re-validating gradients (tests/test_fused_mixer.py)
// before trusting it. Currently covers the repo's real trm_mixer configs: sudoku (36, 64,
// configs/experiments/m23_sudoku_mixer_*.yaml), ETTh1 forecasting (7, 8, m26_etth1_forecast.yaml),
// weather forecasting (21, 8, m26_weather_forecast.yaml), plus (6, 6) for fast unit tests.
#define FOR_EACH_SUPPORTED_SHAPE(X) X(36, 64) X(7, 8) X(21, 8) X(6, 6)

std::vector<torch::Tensor> tm_forward(
    torch::Tensor inp, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2
) {
    TORCH_CHECK(inp.is_cuda() && inp.scalar_type() == torch::kFloat32, "need fp32 CUDA inp");
    auto inp_c = inp.contiguous();
    auto w1_c = w1.contiguous(), w2_c = w2.contiguous();
    auto b1_c = b1.contiguous(), b2_c = b2.contiguous();
    int B = inp_c.size(0), n_cells = inp_c.size(1), in_dim = inp_c.size(2);
    int token_hidden = w1.size(0);

    auto out = torch::empty_like(inp_c);
    auto h   = torch::empty({B, token_hidden, in_dim}, inp_c.options());
    auto pre = torch::empty({B, token_hidden, in_dim}, inp_c.options());

    dim3 grid((in_dim + C_TILE - 1) / C_TILE, B), block(C_TILE);
    auto stream = at::cuda::getCurrentCUDAStream();
    bool ok = false;
#define FWD(NC, TH)                                                                            \
    if (!ok && n_cells == (NC) && token_hidden == (TH)) {                                      \
        tm_fwd<NC, TH, C_TILE><<<grid, block, 0, stream>>>(                                    \
            inp_c.data_ptr<float>(), w1_c.data_ptr<float>(), b1_c.data_ptr<float>(),            \
            w2_c.data_ptr<float>(), b2_c.data_ptr<float>(), out.data_ptr<float>(),              \
            h.data_ptr<float>(), pre.data_ptr<float>(), in_dim);                               \
        ok = true;                                                                             \
    }
    FOR_EACH_SUPPORTED_SHAPE(FWD)
#undef FWD
    TORCH_CHECK(ok, "tm_forward: unsupported (n_cells=", n_cells, ", token_hidden=", token_hidden,
                "); see FOR_EACH_SUPPORTED_SHAPE in fused_token_mix.cu");
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "tm_fwd launch failed");
    return {out, h, pre};
}

std::vector<torch::Tensor> tm_backward(
    torch::Tensor w1, torch::Tensor w2, torch::Tensor pre_h, torch::Tensor grad_out
) {
    TORCH_CHECK(grad_out.is_cuda(), "need CUDA grad_out");
    auto go_c = grad_out.contiguous(), pre_c = pre_h.contiguous();
    auto w1_c = w1.contiguous(), w2_c = w2.contiguous();
    int B = go_c.size(0), n_cells = go_c.size(1), in_dim = go_c.size(2);
    int token_hidden = w1.size(0);

    auto grad_inp = torch::empty_like(go_c);
    auto dpre_h = torch::empty({B, token_hidden, in_dim}, go_c.options());

    dim3 grid((in_dim + C_TILE - 1) / C_TILE, B), block(C_TILE);
    auto stream = at::cuda::getCurrentCUDAStream();
    bool ok = false;
#define BWD(NC, TH)                                                                            \
    if (!ok && n_cells == (NC) && token_hidden == (TH)) {                                      \
        tm_bwd<NC, TH, C_TILE><<<grid, block, 0, stream>>>(                                    \
            w1_c.data_ptr<float>(), w2_c.data_ptr<float>(), pre_c.data_ptr<float>(),            \
            go_c.data_ptr<float>(), grad_inp.data_ptr<float>(), dpre_h.data_ptr<float>(),       \
            in_dim);                                                                           \
        ok = true;                                                                             \
    }
    FOR_EACH_SUPPORTED_SHAPE(BWD)
#undef BWD
    TORCH_CHECK(ok, "tm_backward: unsupported (n_cells=", n_cells, ", token_hidden=", token_hidden,
                "); see FOR_EACH_SUPPORTED_SHAPE in fused_token_mix.cu");
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "tm_bwd launch failed");
    return {grad_inp, dpre_h};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tm_forward", &tm_forward, "Fused token-mix forward (CUDA)");
    m.def("tm_backward", &tm_backward, "Fused token-mix backward (CUDA)");
}
