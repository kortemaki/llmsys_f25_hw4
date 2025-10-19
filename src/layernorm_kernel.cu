#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32


/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = min(hidden_size, MAX_THREADS)

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {

  /// BEGIN ASSIGN4_2_1
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for speedup

  // Step 1
  float l_sums[2] = {0};
  __shared__ float sums[2];
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float4  val = inp_f4[idx];
    l_sums[0] += val.x + val.y + val.z + val.w;
    l_sums[1] += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
  }

  // Step 2
  blockReduce<ReduceType::kSum, 2>(l_sums);
  if (threadIdx.x == 0) {
    sums[0] = l_sums[0];
    sums[1] = l_sums[1];
  }
  __syncthreads();
  const float  mean_x = sums[0] / (hidden_size << 2);
  const float mean_x2 = sums[1] / (hidden_size << 2);
  const float variance = mean_x2 - mean_x * mean_x + LN_EPSILON;
  const float sigma = sqrtf(variance);

  // Step 3
  if (threadIdx.x == 0) {
    if (means) means[blockIdx.x] = mean_x;
    vars[blockIdx.x] = variance;
  }
  float4 *ln_res_f4 = reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  const float4 *scale_f4 = reinterpret_cast<const float4 *>(scale);
  const float4  *bias_f4 = reinterpret_cast<const float4 *>( bias);
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float4 val = inp_f4[idx];
    const float4 scale_i = scale_f4[idx];
    const float4 bias_i = bias_f4[idx];
    ln_res_f4[idx] = make_float4(
      scale_i.x * (val.x - mean_x) / sigma + bias_i.x,
      scale_i.y * (val.y - mean_x) / sigma + bias_i.y,
      scale_i.z * (val.z - mean_x) / sigma + bias_i.z,
      scale_i.w * (val.w - mean_x) / sigma + bias_i.w
    );
  }
  /// END ASSIGN4_2_1
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;


  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);
  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
    d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);

}
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backword kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *gamma,
                                        const T *betta, const T *vars,
                                        const T *means, int rows, int width) {

  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  //      -> More hints about `g.shfl_down`:
  //      -> https://developer.nvidia.com/blog/cooperative-groups/#:~:text=Using%20thread_block_tile%3A%3Ashfl_down()%20to%20simplify%20our%20warp%2Dlevel%20reduction%20does%20benefit%20our%20code%3A%20it%20simplifies%20it%20and%20eliminates%20the%20need%20for%20shared%20memory
  //      -> The highlighted line gives you a conceptual understanding of what the g.shfl_down is doing. Usually, the threads inside a block need to load everything to shared memory and work together to reduce the result (like what you have implemented in the hw1 for reduce function).
  //      -> Now g.shfl_down helps you do so without consuming any shared memory. g.shfl_down makes it more efficient.
  // 4. Assign the final result to the correct position in the global output

  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];

  if (!vars || !means) {
    assert(false && "Error: invalid input! Both vars and means must be provided.");
  }
  if (blockIdx.y) return;

  const uint idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  const uint idx_y = threadIdx.y;
  const uint size = rows * width;
  inp += idx_y * width;
  out_grad += idx_y * width;

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  // Step 1
  const uint stride = blockDim.y * width;
  uint i = idx_x;
  T l_d_gam = 0;
  T l_d_bet = 0;
  for (uint i_y = idx_y; i_y < rows; i_y += blockDim.y) {
    if (i >= size) break;
    l_d_gam += out_grad[i] * (inp[i] - means[i_y]) * rsqrt(vars[i_y] + LN_EPSILON);
    l_d_bet += out_grad[i];
    i += stride;
  }

  // Step 2
  betta_buffer[threadIdx.x][threadIdx.y] = l_d_bet;
  gamma_buffer[threadIdx.x][threadIdx.y] = l_d_gam;
  __syncthreads();

  // Step 3
  l_d_bet = betta_buffer[threadIdx.y][threadIdx.x];
  l_d_gam = gamma_buffer[threadIdx.y][threadIdx.x];
  for (int i = g.size() / 2; i > 0; i /= 2) {
    l_d_gam += g.shfl_down(l_d_gam, i);
    l_d_bet += g.shfl_down(l_d_bet, i);
  }
  if (!threadIdx.x) {
    betta_buffer[threadIdx.y][threadIdx.x] = l_d_bet;
    gamma_buffer[threadIdx.y][threadIdx.x] = l_d_gam;
  }
  __syncthreads();

  // Step 4
  if (idx_y) return;
  if (idx_x >= width) return;
  gamma_grad[idx_x] = gamma_buffer[threadIdx.x][threadIdx.y];
  betta_grad[idx_x] = betta_buffer[threadIdx.x][threadIdx.y];
  /// END ASSIGN4_2_2
}

/**
@brief: ker_ln_bw_dinp
Layer norm backword kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {

  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute dxhat=dy*w with reinterpret_cast by casting to float4 for speedup
  // 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  // 3. Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  // 4. Compute final gradient

  if (!vars || !means) {
    assert(false && "Error: invalid input! Both vars and means must be provided.");
  }
  if (threadIdx.x >= hidden_dim) return;

  const uint idx_y = blockIdx.x;
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + idx_y * hidden_dim;
  const T mean = means[idx_y];
  const T rstd = rsqrt(vars[idx_y] + LN_EPSILON);
  const float4 *out_grad_f4 = reinterpret_cast<const float4 *>(out_grad) + idx_y * hidden_dim;
  const float4 *gamma_f4 = reinterpret_cast<const float4 *>(gamma);
  float4 *inp_grad_f4 = reinterpret_cast<float4 *>(inp_grad) + idx_y * hidden_dim;

  const uint idx = threadIdx.x;
  float4 dxhat;
  float4 xhat;
  float l_sums[2];
  __shared__ float sums[2];

  const float4 y_j = out_grad_f4[idx];
  const float4 gamma_j = gamma_f4[idx];
  dxhat.x = y_j.x * gamma_j.x;
  dxhat.y = y_j.y * gamma_j.y;
  dxhat.z = y_j.z * gamma_j.z;
  dxhat.w = y_j.w * gamma_j.w;

  // Step 2
  const float4 inp_j = inp_f4[idx];
  xhat.x = (inp_j.x - mean) * rstd;
  xhat.y = (inp_j.y - mean) * rstd;
  xhat.z = (inp_j.z - mean) * rstd;
  xhat.w = (inp_j.w - mean) * rstd;

  // Step 3
  l_sums[0] = dxhat.x + dxhat.y + dxhat.z + dxhat.w;
  l_sums[1] = xhat.x * dxhat.x + xhat.y * dxhat.y + xhat.z * dxhat.z + xhat.w * dxhat.w;

  blockReduce<ReduceType::kSum, 2>(l_sums);
  if (!threadIdx.x) {
    sums[0] = l_sums[0];
    sums[1] = l_sums[1];
    printf("sums row %d: dxhat %f xhat_dxhat %f\n", blockIdx.x, sums[0], sums[1]);
  }
  __syncthreads();
  printf("thread %d %d xhat [%f, %f, %f, %f] dxhat [%f, %f, %f, %f]\n", blockIdx.x, threadIdx.x, xhat.x, xhat.y, xhat.z, xhat.w, dxhat.x, dxhat.y, dxhat.z, dxhat.w);

  // Step 4
  // divide both sums by m here to save repeated divides below
  uint m = hidden_dim << 2;
  float sum_dxhat_m = sums[0] / m;
  float sum_xhat_dxhat_m = sums[1] / m;
  inp_grad_f4[idx] = make_float4(
    (dxhat.x - sum_dxhat_m - xhat.x * sum_xhat_dxhat_m) * rstd,
    (dxhat.y - sum_dxhat_m - xhat.y * sum_xhat_dxhat_m) * rstd,
    (dxhat.z - sum_dxhat_m - xhat.z * sum_xhat_dxhat_m) * rstd,
    (dxhat.w - sum_dxhat_m - xhat.w * sum_xhat_dxhat_m) * rstd
  );
  /// END ASSIGN4_2_2
}
// WARNING: because of the way the loops below have been unrolled, this needs to be launched with the exact iteration count.
template <typename T, int ITERATIONS>
__global__ void ker_ln_bw_dinp_gt4096(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {
  // Here we will allocate dynamic memory for xhat and dxhat on the stack
  // The stack is fairly generous but it is limited, so we probably wouldn't want to use this kernel with iterations > 1000 or so
  float4 xhat[ITERATIONS];
  float4 dxhat[ITERATIONS];

  if (!vars || !means) {
    assert(false && "Error: invalid input! Both vars and means must be provided.");
  }

  const uint idx_y = blockIdx.x;
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + idx_y * hidden_dim;
  const T mean = means[idx_y];
  const T rstd = rsqrt(vars[idx_y] + LN_EPSILON);
  const float4 *out_grad_f4 = reinterpret_cast<const float4 *>(out_grad) + idx_y * hidden_dim;
  const float4 *gamma_f4 = reinterpret_cast<const float4 *>(gamma);
  float4 *inp_grad_f4 = reinterpret_cast<float4 *>(inp_grad) + idx_y * hidden_dim;

  const uint idx = threadIdx.x;
  float l_sums[2];
  __shared__ float sums[2];
  uint i = idx;
  uint k = 0;
  #pragma unroll
  for (; k < ITERATIONS - 1; k++) {
    // Step 1
    const float4 y_j = out_grad_f4[i];
    const float4 gamma_j = gamma_f4[i];
    dxhat[k] = make_float4(
      y_j.x * gamma_j.x,
      y_j.y * gamma_j.y,
      y_j.z * gamma_j.z,
      y_j.w * gamma_j.w
    );

    // Step 2
    const float4 inp_j = inp_f4[i];
    xhat[k] = make_float4(
      (inp_j.x - mean) * rstd,
      (inp_j.y - mean) * rstd,
      (inp_j.z - mean) * rstd,
      (inp_j.w - mean) * rstd
    );

    // Step 3
    l_sums[0] = dxhat[k].x + dxhat[k].y + dxhat[k].z + dxhat[k].w;
    l_sums[1] = xhat[k].x * dxhat[k].x + xhat[k].y * dxhat[k].y + xhat[k].z * dxhat[k].z + xhat[k].w * dxhat[k].w;
    i += blockDim.x;
  }
  // manually unroll the last iteration to avoid a branch in the loop
  if (i < hidden_dim) {
    // Step 1
    const float4 y_j = out_grad_f4[i];
    const float4 gamma_j = gamma_f4[i];
    dxhat[k] = make_float4(
      y_j.x * gamma_j.x,
      y_j.y * gamma_j.y,
      y_j.z * gamma_j.z,
      y_j.w * gamma_j.w
    );

    // Step 2
    const float4 inp_j = inp_f4[i];
    xhat[k] = make_float4(
      (inp_j.x - mean) * rstd,
      (inp_j.y - mean) * rstd,
      (inp_j.z - mean) * rstd,
      (inp_j.w - mean) * rstd
    );

    // Step 3
    l_sums[0] = dxhat[k].x + dxhat[k].y + dxhat[k].z + dxhat[k].w;
    l_sums[1] = xhat[k].x * dxhat[k].x + xhat[k].y * dxhat[k].y + xhat[k].z * dxhat[k].z + xhat[k].w * dxhat[k].w;
  }

  blockReduce<ReduceType::kSum, 2>(l_sums);
  if (!threadIdx.x) {
    sums[0] = l_sums[0];
    sums[1] = l_sums[1];
  }
  __syncthreads();

  // Step 4
  // divide both sums by m here to save repeated divides below
  uint m = hidden_dim << 2;
  float sum_dxhat_m = sums[0] / m;
  float sum_xhat_dxhat_m = sums[1] / m;
  i = idx;
  k = 0;
  #pragma unroll
  for (; k < ITERATIONS - 1; k++) {
    float4 inp_grad_i = make_float4(
      (dxhat[k].x - sum_dxhat_m - xhat[k].x * sum_xhat_dxhat_m) * rstd,
      (dxhat[k].y - sum_dxhat_m - xhat[k].y * sum_xhat_dxhat_m) * rstd,
      (dxhat[k].z - sum_dxhat_m - xhat[k].z * sum_xhat_dxhat_m) * rstd,
      (dxhat[k].w - sum_dxhat_m - xhat[k].w * sum_xhat_dxhat_m) * rstd
    );
    inp_grad_f4[i] = inp_grad_i;
    i += blockDim.x;
  }
  // manually unroll the last loop because of early returns
  if (i >= hidden_dim) return;
  float4 inp_grad_i = make_float4(
    (dxhat[k].x - sum_dxhat_m - xhat[k].x * sum_xhat_dxhat_m) * rstd,
    (dxhat[k].y - sum_dxhat_m - xhat[k].y * sum_xhat_dxhat_m) * rstd,
    (dxhat[k].z - sum_dxhat_m - xhat[k].z * sum_xhat_dxhat_m) * rstd,
    (dxhat[k].w - sum_dxhat_m - xhat[k].w * sum_xhat_dxhat_m) * rstd
  );
  inp_grad_f4[i] = inp_grad_i;
  /// END ASSIGN4_2_2
}
template <typename T>
__global__ void ker_ln_bw_dinp_gt16384(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {
  // Here we will allocate dynamic memory for xhat and dxhat on the heap
  // This should be large enough that we will have other problems if we run out
  const uint ITERATIONS = hidden_dim / MAX_THREADS;
  float4 *xhat = (float4*) malloc(ITERATIONS * sizeof(float4));
  float4 *dxhat = (float4*) malloc(ITERATIONS * sizeof(float4));

  if (!vars || !means) {
    assert(false && "Error: invalid input! Both vars and means must be provided.");
  }

  const uint idx_y = blockIdx.x;
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + idx_y * hidden_dim;
  const T mean = means[idx_y];
  const T rstd = rsqrt(vars[idx_y] + LN_EPSILON);
  const float4 *out_grad_f4 = reinterpret_cast<const float4 *>(out_grad) + idx_y * hidden_dim;
  const float4 *gamma_f4 = reinterpret_cast<const float4 *>(gamma);
  float4 *inp_grad_f4 = reinterpret_cast<float4 *>(inp_grad) + idx_y * hidden_dim;

  const uint idx = threadIdx.x;
  float l_sums[2];
  __shared__ float sums[2];
  uint k = 0;
  for (uint i = idx; i < hidden_dim; i += blockDim.x) {
    // Step 1
    const float4 y_j = out_grad_f4[i];
    const float4 gamma_j = gamma_f4[i];
    dxhat[k] = make_float4(
      y_j.x * gamma_j.x,
      y_j.y * gamma_j.y,
      y_j.z * gamma_j.z,
      y_j.w * gamma_j.w
    );

    // Step 2
    const float4 inp_j = inp_f4[i];
    xhat[k] = make_float4(
      (inp_j.x - mean) * rstd,
      (inp_j.y - mean) * rstd,
      (inp_j.z - mean) * rstd,
      (inp_j.w - mean) * rstd
    );

    // Step 3
    l_sums[0] = dxhat[k].x + dxhat[k].y + dxhat[k].z + dxhat[k].w;
    l_sums[1] = xhat[k].x * dxhat[k].x + xhat[k].y * dxhat[k].y + xhat[k].z * dxhat[k].z + xhat[k].w * dxhat[k].w;
    k += 1;
  }

  blockReduce<ReduceType::kSum, 2>(l_sums);
  if (!threadIdx.x) {
    sums[0] = l_sums[0];
    sums[1] = l_sums[1];
  }
  __syncthreads();

  // Step 4
  // divide both sums by m here to save repeated divides below
  uint m = hidden_dim << 2;
  float sum_dxhat_m = sums[0] / m;
  float sum_xhat_dxhat_m = sums[1] / m;
  k = 0;
  for (uint i = idx; i < hidden_dim; i += blockDim.x) {
    float4 inp_grad_i = make_float4(
      (dxhat[k].x - sum_dxhat_m - xhat[k].x * sum_xhat_dxhat_m) * rstd,
      (dxhat[k].y - sum_dxhat_m - xhat[k].y * sum_xhat_dxhat_m) * rstd,
      (dxhat[k].z - sum_dxhat_m - xhat[k].z * sum_xhat_dxhat_m) * rstd,
      (dxhat[k].w - sum_dxhat_m - xhat[k].w * sum_xhat_dxhat_m) * rstd
    );
    inp_grad_f4[i] = inp_grad_i;
    k += 1;
  }
  free(xhat);
  free(dxhat);
  /// END ASSIGN4_2_2
}
extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp, const float *gamma,
                         const float *betta, const float *vars,
                         const float *means, int batch_size, int hidden_dim,
                         cudaStream_t stream_1, cudaStream_t stream_2) {

  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_betta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
  dim3 grid_dim((hidden_dim + TILE_DIM - 1) / TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);

  // Compute grad of input
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("hidden_dim % 4 != 0");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  if (hidden_dim <= 4096) {
    ker_ln_bw_dinp<float><<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);
  } else if (hidden_dim <= 8192) {
    ker_ln_bw_dinp_gt4096<float, 2><<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);
  } else if (hidden_dim <= 12288) {
    ker_ln_bw_dinp_gt4096<float, 3><<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);
  } else if (hidden_dim <= 16384) {
    ker_ln_bw_dinp_gt4096<float, 4><<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);
  } else {
    ker_ln_bw_dinp_gt16384<float><<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);
  }

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}}
}}
