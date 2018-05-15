//pass
//--gridDim=6400  --blockDim=64 --warp-sync=32

typedef unsigned int uint;
typedef unsigned short ushort;
#define FLT_MAX 0x1.fffffep127f

                   __device__ static __attribute__((always_inline)) void colorSums(const float3 *colors, float3 *sums);
                   __device__ static __attribute__((always_inline)) float3 firstEigenVector(float matrix[6]);
                   __device__ static __attribute__((always_inline)) float3 bestFitLine(const float3 *colors, float3 color_sum);
template <class T> __device__ static __attribute__((always_inline)) void swap(T &a, T &b);
                   __device__ static __attribute__((always_inline)) void sortColors(const float *values, int *ranks);
                   __device__ static __attribute__((always_inline)) void loadColorBlock(const uint *image, float3 colors[16], float3 sums[16], int xrefs[16], int blockOffset);
                   __device__ static __attribute__((always_inline)) float3 roundAndExpand(float3 v, ushort *w);
                   __device__ static __attribute__((always_inline)) float evalPermutation4(const float3 *colors, uint permutation, ushort *start, ushort *end, float3 color_sum);
                   __device__ static __attribute__((always_inline)) float evalPermutation3(const float3 *colors, uint permutation, ushort *start, ushort *end, float3 color_sum);
                   __device__ static __attribute__((always_inline)) void evalAllPermutations(const float3 *colors, const uint *permutations, ushort &bestStart, ushort &bestEnd, uint &bestPermutation, float *errors, float3 color_sum);
                   __device__ static __attribute__((always_inline)) int findMinError(float *errors);
                   __device__ static __attribute__((always_inline)) void saveBlockDXT1(ushort start, ushort end, uint permutation, int xrefs[16], uint2 *result, int blockOffset);

#define NUM_THREADS 64        // Number of threads per block.

__device__ static __attribute__((always_inline)) void colorSums(const float3 *colors, float3 *sums)
{
    const int idx = threadIdx.x;

    sums[idx] = colors[idx];
    sums[idx] += sums[idx^8];
    sums[idx] += sums[idx^4];
    sums[idx] += sums[idx^2];
    sums[idx] += sums[idx^1];
}

__device__ static __attribute__((always_inline)) float3 firstEigenVector(float matrix[6])
{
    // 8 iterations seems to be more than enough.

    float3 v = make_float3(1.0f, 1.0f, 1.0f);

    for (int i = 0;
         __global_invariant(__implies(threadIdx.x >= 16, !__enabled())),
         i < 8; i++)
    {
        float x = v.x * matrix[0] + v.y * matrix[1] + v.z * matrix[2];
        float y = v.x * matrix[1] + v.y * matrix[3] + v.z * matrix[4];
        float z = v.x * matrix[2] + v.y * matrix[4] + v.z * matrix[5];
        float m = max(max(x, y), z);
        float iv = 1.0f / m;
        v = make_float3(x*iv, y*iv, z*iv);
    }

    return v;
}

__device__ static __attribute__((always_inline)) float3 bestFitLine(const float3 *colors, float3 color_sum)
{
    // Compute covariance matrix of the given colors.
    const int idx = threadIdx.x;

    float3 diff = colors[idx] - color_sum * (1.0f / 16.0f);

    // @@ Eliminate two-way bank conflicts here.
    // @@ It seems that doing that and unrolling the reduction doesn't help...
    __shared__ float covariance[16*6];

    covariance[6 * idx + 0] = diff.x * diff.x;    // 0, 6, 12, 2, 8, 14, 4, 10, 0
    covariance[6 * idx + 1] = diff.x * diff.y;
    covariance[6 * idx + 2] = diff.x * diff.z;
    covariance[6 * idx + 3] = diff.y * diff.y;
    covariance[6 * idx + 4] = diff.y * diff.z;
    covariance[6 * idx + 5] = diff.z * diff.z;

    for (int d = 8;
         __global_invariant(__implies(idx >= 16, !__enabled())),
         __global_invariant(__implies(idx >= 16, !__write(covariance))),
         __global_invariant(__implies(idx >= 16, !__read(covariance))),
         __global_invariant(__implies(idx/32 == __other_int(idx)/32 & blockIdx.x == __other_int(blockIdx.x), !__write(covariance))),
         d > 0; d >>= 1)
    {
        if (idx < d)
        {
            covariance[6 * idx + 0] += covariance[6 * (idx+d) + 0];
            covariance[6 * idx + 1] += covariance[6 * (idx+d) + 1];
            covariance[6 * idx + 2] += covariance[6 * (idx+d) + 2];
            covariance[6 * idx + 3] += covariance[6 * (idx+d) + 3];
            covariance[6 * idx + 4] += covariance[6 * (idx+d) + 4];
            covariance[6 * idx + 5] += covariance[6 * (idx+d) + 5];
        }
    }

    // Compute first eigen vector.
    return firstEigenVector(covariance);
}

template <class T>
__device__ static __attribute__((always_inline)) void swap(T &a, T &b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

//__constant__ float3 kColorMetric = { 0.2126f, 0.7152f, 0.0722f };
__constant__ float3 kColorMetric = { 1.0f, 1.0f, 1.0f };

__device__ static __attribute__((always_inline)) void sortColors(const float *values, int *ranks)
{
    const int tid = threadIdx.x;

    int rank = 0;

#pragma unroll

    for (int i = 0;
         __global_invariant(__implies(tid >= 16, !__enabled())),
         i < 16; i++)
    {
        rank += (values[i] < values[tid]);
    }

    ranks[tid] = rank;

    // Resolve elements with the same index.
#pragma unroll

    for (int i = 0;
         __global_invariant(__implies(tid >= 16, !__enabled())),
         __global_invariant(__implies(tid >= 16, !__read(ranks))),
         __global_invariant(__implies(tid/32 == __other_int(tid)/32 & blockIdx.x == __other_int(blockIdx.x), !__write(ranks))),
         i < 15; i++)
    {
        if (tid > i && ranks[tid] == ranks[i])
        {
            ++ranks[tid];
        }
    }

    // IMPERIAL EDIT: post condition of the above code
    __assume(__implies(tid < 16 & __other_int(tid) < 16 & blockIdx.x == __other_int(blockIdx.x), ranks[tid] != ranks[__other_int(tid)]));
}

__device__ static __attribute__((always_inline)) void loadColorBlock(const uint *image, float3 colors[16], float3 sums[16], int xrefs[16], int blockOffset)
{
    const int bid = blockIdx.x + blockOffset;
    const int idx = threadIdx.x;

    __shared__ float dps[16];

    float3 tmp;

    if (idx < 16)
    {
        // Read color and copy to shared mem.
        uint c = image[(bid) * 16 + idx];

        colors[idx].x = ((c >> 0) & 0xFF) * (1.0f / 255.0f);
        colors[idx].y = ((c >> 8) & 0xFF) * (1.0f / 255.0f);
        colors[idx].z = ((c >> 16) & 0xFF) * (1.0f / 255.0f);

        // Sort colors along the best fit line.
        colorSums(colors, sums);
        float3 axis = bestFitLine(colors, sums[0]);

        dps[idx] = dot(colors[idx], axis);

        sortColors(dps, xrefs);

        tmp = colors[idx];

        colors[xrefs[idx]] = tmp;
    }
}

__device__ static __attribute__((always_inline)) float3 roundAndExpand(float3 v, ushort *w)
{
    v.x = rintf(__saturatef(v.x) * 31.0f);
    v.y = rintf(__saturatef(v.y) * 63.0f);
    v.z = rintf(__saturatef(v.z) * 31.0f);

    *w = ((ushort)v.x << 11) | ((ushort)v.y << 5) | (ushort)v.z;
    v.x *= 0.03227752766457f; // approximate integer bit expansion.
    v.y *= 0.01583151765563f;
    v.z *= 0.03227752766457f;
    return v;
}

__constant__ float alphaTable4[4] = { 9.0f, 0.0f, 6.0f, 3.0f };
__constant__ float alphaTable3[4] = { 4.0f, 0.0f, 2.0f, 2.0f };
__constant__ const int prods4[4] = { 0x090000,0x000900,0x040102,0x010402 };
__constant__ const int prods3[4] = { 0x040000,0x000400,0x040101,0x010401 };

#define USE_TABLES 1

__device__ static __attribute__((always_inline)) float evalPermutation4(const float3 *colors, uint permutation, ushort *start, ushort *end, float3 color_sum)
{
    // Compute endpoints using least squares.
#if USE_TABLES
    float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

    int akku = 0;

    // Compute alpha & beta for this permutation.
    for (int i = 0; i < 16; i++)
    {
        const uint bits = permutation >> (2*i);

        alphax_sum += alphaTable4[bits & 3] * colors[i];
        akku += prods4[bits & 3];
    }

    float alpha2_sum = float(akku >> 16);
    float beta2_sum = float((akku >> 8) & 0xff);
    float alphabeta_sum = float((akku >> 0) & 0xff);
    float3 betax_sum = (9.0f * color_sum) - alphax_sum;
#else
    float alpha2_sum = 0.0f;
    float beta2_sum = 0.0f;
    float alphabeta_sum = 0.0f;
    float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

    // Compute alpha & beta for this permutation.
    for (int i = 0; i < 16; i++)
    {
        const uint bits = permutation >> (2*i);

        float beta = (bits & 1);

        if (bits & 2)
        {
            beta = (1 + beta) * (1.0f / 3.0f);
        }

        float alpha = 1.0f - beta;

        alpha2_sum += alpha * alpha;
        beta2_sum += beta * beta;
        alphabeta_sum += alpha * beta;
        alphax_sum += alpha * colors[i];
    }

    float3 betax_sum = color_sum - alphax_sum;
#endif

    // alpha2, beta2, alphabeta and factor could be precomputed for each permutation, but it's faster to recompute them.
    const float factor = 1.0f / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

    float3 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
    float3 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;

    // Round a, b to the closest 5-6-5 color and expand...
    a = roundAndExpand(a, start);
    b = roundAndExpand(b, end);

    // compute the error
    float3 e = a * a * alpha2_sum + b * b * beta2_sum + 2.0f * (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum);

    return (0.111111111111f) * dot(e, kColorMetric);
}

__device__ static __attribute__((always_inline)) float evalPermutation3(const float3 *colors, uint permutation, ushort *start, ushort *end, float3 color_sum)
{
    // Compute endpoints using least squares.
#if USE_TABLES
    float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

    int akku = 0;

    // Compute alpha & beta for this permutation.
    for (int i = 0; i < 16; i++)
    {
        const uint bits = permutation >> (2*i);

        alphax_sum += alphaTable3[bits & 3] * colors[i];
        akku += prods3[bits & 3];
    }

    float alpha2_sum = float(akku >> 16);
    float beta2_sum = float((akku >> 8) & 0xff);
    float alphabeta_sum = float((akku >> 0) & 0xff);
    float3 betax_sum = (4.0f * color_sum) - alphax_sum;
#else
    float alpha2_sum = 0.0f;
    float beta2_sum = 0.0f;
    float alphabeta_sum = 0.0f;
    float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

    // Compute alpha & beta for this permutation.
    for (int i = 0; i < 16; i++)
    {
        const uint bits = permutation >> (2*i);

        float beta = (bits & 1);

        if (bits & 2)
        {
            beta = 0.5f;
        }

        float alpha = 1.0f - beta;

        alpha2_sum += alpha * alpha;
        beta2_sum += beta * beta;
        alphabeta_sum += alpha * beta;
        alphax_sum += alpha * colors[i];
    }

    float3 betax_sum = color_sum - alphax_sum;
#endif

    const float factor = 1.0f / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

    float3 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
    float3 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;

    // Round a, b to the closest 5-6-5 color and expand...
    a = roundAndExpand(a, start);
    b = roundAndExpand(b, end);

    // compute the error
    float3 e = a * a * alpha2_sum + b * b * beta2_sum + 2.0f * (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum);

    return (0.25f) * dot(e, kColorMetric);
}

__device__ static __attribute__((always_inline)) void evalAllPermutations(const float3 *colors, const uint *permutations, ushort &bestStart, ushort &bestEnd, uint &bestPermutation, float *errors, float3 color_sum)
{
    const int idx = threadIdx.x;

    float bestError = FLT_MAX;

    __shared__ uint s_permutations[160];

    for (int i = 0; i < 16; i++)
    {
        int pidx = idx + NUM_THREADS * i;

        if (pidx >= 992)
        {
            break;
        }

        ushort start, end;
        uint permutation = permutations[pidx];

        if (pidx < 160)
        {
            s_permutations[pidx] = permutation;
        }

        float error = evalPermutation4(colors, permutation, &start, &end, color_sum);

        if (error < bestError)
        {
            bestError = error;
            bestPermutation = permutation;
            bestStart = start;
            bestEnd = end;
        }
    }

    if (bestStart < bestEnd)
    {
        swap(bestEnd, bestStart);
        bestPermutation ^= 0x55555555;    // Flip indices.
    }

    for (int i = 0; i < 3; i++)
    {
        int pidx = idx + NUM_THREADS * i;

        if (pidx >= 160)
        {
            break;
        }

        ushort start, end;
        uint permutation = s_permutations[pidx];
        float error = evalPermutation3(colors, permutation, &start, &end, color_sum);

        if (error < bestError)
        {
            bestError = error;
            bestPermutation = permutation;
            bestStart = start;
            bestEnd = end;

            if (bestStart > bestEnd)
            {
                swap(bestEnd, bestStart);
                bestPermutation ^= (~bestPermutation >> 1) & 0x55555555;    // Flip indices.
            }
        }
    }

    errors[idx] = bestError;
}

__device__ static __attribute__((always_inline)) int findMinError(float *errors)
{
    const int idx = threadIdx.x;

    __shared__ int indices[NUM_THREADS];
    indices[idx] = idx;

    for (int d = NUM_THREADS/2; d > 32; d >>= 1)
    {
        __syncthreads();

        if (idx < d)
        {
            float err0 = errors[idx];
            float err1 = errors[idx + d];

            if (err1 < err0)
            {
                errors[idx] = err1;
                indices[idx] = indices[idx + d];
            }
        }
    }

    __syncthreads();

    // unroll last 6 iterations
    if (idx < 32)
    {
        if (errors[idx + 32] < errors[idx])
        {
            errors[idx] = errors[idx + 32];
            indices[idx] = indices[idx + 32];
        }

        if (errors[idx + 16] < errors[idx])
        {
            errors[idx] = errors[idx + 16];
            indices[idx] = indices[idx + 16];
        }

        if (errors[idx + 8] < errors[idx])
        {
            errors[idx] = errors[idx + 8];
            indices[idx] = indices[idx + 8];
        }

        if (errors[idx + 4] < errors[idx])
        {
            errors[idx] = errors[idx + 4];
            indices[idx] = indices[idx + 4];
        }

        if (errors[idx + 2] < errors[idx])
        {
            errors[idx] = errors[idx + 2];
            indices[idx] = indices[idx + 2];
        }

        if (errors[idx + 1] < errors[idx])
        {
            errors[idx] = errors[idx + 1];
            indices[idx] = indices[idx + 1];
        }
    }

    __syncthreads();

    return indices[0];
}

__device__ static __attribute__((always_inline)) void saveBlockDXT1(ushort start, ushort end, uint permutation, int xrefs[16], uint2 *result, int blockOffset)
{
    const int bid = blockIdx.x + blockOffset;

    if (start == end)
    {
        permutation = 0;
    }

    // Reorder permutation.
    uint indices = 0;

    for (int i = 0; i < 16; i++)
    {
        int ref = xrefs[i];
        indices |= ((permutation >> (2 * ref)) & 3) << (2 * i);
    }

    // Write endpoints.
    result[bid].x = (end << 16) | start;

    // Write palette indices.
    result[bid].y = indices;
}

__global__ void compress(const uint *permutations, const uint *image, uint2 *result, int blockOffset)
{
    const int idx = threadIdx.x;

    __shared__ float3 colors[16];
    __shared__ float3 sums[16];
    __shared__ int xrefs[16];

    loadColorBlock(image, colors, sums, xrefs, blockOffset);

    __syncthreads();

    ushort bestStart, bestEnd;
    uint bestPermutation;

    __shared__ float errors[NUM_THREADS];

    evalAllPermutations(colors, permutations, bestStart, bestEnd, bestPermutation, errors, sums[0]);

    // Use a parallel reduction to find minimum error.
    const int minIdx = findMinError(errors);

    __syncthreads();

    // Only write the result of the winner thread.
    if (idx == minIdx)
    {
        saveBlockDXT1(bestStart, bestEnd, bestPermutation, xrefs, result, blockOffset);
    }
}
