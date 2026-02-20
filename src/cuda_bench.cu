/*
 * FrameForge — cuda_bench.cu
 * ==========================
 * C++ CUDA shader-simulation benchmarks with NVTX profiling annotations.
 * Simulates real GPU gaming pipeline stages: vertex transform, pixel shading,
 * post-processing blur, and texture sampling.
 *
 * Compile:  nvcc -O2 -arch=sm_60 -lnvToolsExt -o cuda_bench cuda_bench.cu
 * Run:      ./cuda_bench
 * Profile:  nsys profile --trace=cuda,nvtx -o cuda_bench_profile ./cuda_bench
 *
 * Author: Akshay Keerthi AS
 * Date: Feb 2026
 */

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// =============================================================================
// Error checking macro
// =============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// =============================================================================
// Configuration
// =============================================================================
typedef struct {
    const char* name;
    int width;
    int height;
} Resolution;

static Resolution RESOLUTIONS[] = {
    {"1080p", 1920, 1080},
    {"1440p", 2560, 1440},
    {"4K",    3840, 2160}
};
static const int NUM_RESOLUTIONS = 3;
static const int WARMUP_ITERS = 5;
static const int BENCH_ITERS = 50;

// =============================================================================
// KERNEL 1: Vertex Shader Simulation
// MVP matrix transform per vertex (4x4 matrix × vec4)
// =============================================================================
__global__ void vertexShaderKernel(float4* vertices, const float* mvpMatrix, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float4 v = vertices[idx];
        float4 result;
        result.x = mvpMatrix[0]*v.x + mvpMatrix[1]*v.y + mvpMatrix[2]*v.z + mvpMatrix[3]*v.w;
        result.y = mvpMatrix[4]*v.x + mvpMatrix[5]*v.y + mvpMatrix[6]*v.z + mvpMatrix[7]*v.w;
        result.z = mvpMatrix[8]*v.x + mvpMatrix[9]*v.y + mvpMatrix[10]*v.z + mvpMatrix[11]*v.w;
        result.w = mvpMatrix[12]*v.x + mvpMatrix[13]*v.y + mvpMatrix[14]*v.z + mvpMatrix[15]*v.w;
        vertices[idx] = result;
    }
}

// =============================================================================
// KERNEL 2: Pixel Shader Simulation (Blinn-Phong Lighting)
// Per-pixel diffuse + specular lighting calculation
// =============================================================================
__global__ void pixelShaderKernel(float* framebuffer, const float* normals,
                                   float lx, float ly, float lz, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H) {
        int idx = (y * W + x) * 3;
        float nx = normals[idx], ny = normals[idx+1], nz = normals[idx+2];

        // Diffuse (Lambert)
        float diff = fmaxf(nx*lx + ny*ly + nz*lz, 0.0f);

        // Specular (Blinn-Phong)
        float hx = lx, hy = ly + 1.0f, hz = lz;
        float invLen = rsqrtf(hx*hx + hy*hy + hz*hz);
        hx *= invLen; hy *= invLen; hz *= invLen;
        float spec = powf(fmaxf(nx*hx + ny*hy + nz*hz, 0.0f), 32.0f);

        // Ambient + Diffuse + Specular
        framebuffer[idx]   = fminf(0.05f + diff * 0.7f + spec * 0.4f, 1.0f);
        framebuffer[idx+1] = fminf(0.05f + diff * 0.5f + spec * 0.4f, 1.0f);
        framebuffer[idx+2] = fminf(0.05f + diff * 0.3f + spec * 0.4f, 1.0f);
    }
}

// =============================================================================
// KERNEL 3: Post-Processing — 5×5 Gaussian Blur
// Simulates common post-FX pass (bloom, DoF, motion blur)
// =============================================================================
__global__ void postFxBlurKernel(const float* input, float* output, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 2 && x < W-2 && y >= 2 && y < H-2) {
        // 5×5 Gaussian weights (sigma ≈ 1.0)
        const float k[5] = {0.06136f, 0.24477f, 0.38774f, 0.24477f, 0.06136f};

        for (int c = 0; c < 3; c++) {
            float sum = 0.0f;
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int si = ((y+ky) * W + (x+kx)) * 3 + c;
                    sum += input[si] * k[ky+2] * k[kx+2];
                }
            }
            output[(y * W + x) * 3 + c] = sum;
        }
    }
}

// =============================================================================
// KERNEL 4: Texture Sampling — Bilinear Interpolation
// Simulates texture fetch with bilinear filtering
// =============================================================================
__global__ void textureSampleKernel(const float* texture, float* output,
                                     const float* uvs, int texW, int texH, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float u = uvs[idx * 2] * (texW - 1);
        float v = uvs[idx * 2 + 1] * (texH - 1);

        int u0 = (int)u, v0 = (int)v;
        int u1 = min(u0 + 1, texW - 1);
        int v1 = min(v0 + 1, texH - 1);
        float fu = u - u0, fv = v - v0;

        for (int c = 0; c < 3; c++) {
            float c00 = texture[(v0 * texW + u0) * 3 + c];
            float c10 = texture[(v0 * texW + u1) * 3 + c];
            float c01 = texture[(v1 * texW + u0) * 3 + c];
            float c11 = texture[(v1 * texW + u1) * 3 + c];
            output[idx * 3 + c] = (1-fu)*(1-fv)*c00 + fu*(1-fv)*c10 +
                                   (1-fu)*fv*c01 + fu*fv*c11;
        }
    }
}

// =============================================================================
// Benchmark Runner
// =============================================================================
typedef struct {
    const char* kernel_name;
    const char* resolution;
    float avg_ms;
    float min_ms;
    float max_ms;
    float throughput;
    const char* throughput_unit;
} BenchResult;

static BenchResult g_results[64];
static int g_numResults = 0;

// Fill array with random floats [0, 1)
void fillRandom(float* arr, int n) {
    for (int i = 0; i < n; i++)
        arr[i] = (float)rand() / (float)RAND_MAX;
}

// Fill with random unit normals
void fillRandomNormals(float* arr, int n) {
    for (int i = 0; i < n; i += 3) {
        float x = (float)rand()/RAND_MAX - 0.5f;
        float y = (float)rand()/RAND_MAX - 0.5f;
        float z = (float)rand()/RAND_MAX - 0.5f;
        float len = sqrtf(x*x + y*y + z*z);
        if (len > 0.001f) { arr[i]=x/len; arr[i+1]=y/len; arr[i+2]=z/len; }
        else { arr[i]=0; arr[i+1]=1; arr[i+2]=0; }
    }
}

void benchVertexShader(Resolution res) {
    nvtxRangePushA("Vertex Shader Benchmark");

    int numVerts = (res.width * res.height) / 4;
    size_t vertBytes = numVerts * sizeof(float4);
    size_t matBytes = 16 * sizeof(float);

    // Host allocation
    nvtxRangePushA("Host Memory Allocation");
    float4* h_verts = (float4*)malloc(vertBytes);
    float* h_mat = (float*)malloc(matBytes);
    for (int i = 0; i < numVerts; i++) {
        h_verts[i] = make_float4((float)rand()/RAND_MAX, (float)rand()/RAND_MAX,
                                  (float)rand()/RAND_MAX, 1.0f);
    }
    // Identity-ish MVP matrix
    memset(h_mat, 0, matBytes);
    h_mat[0] = 1.0f; h_mat[5] = 1.0f; h_mat[10] = 1.0f; h_mat[15] = 1.0f;
    nvtxRangePop();

    // Device allocation
    nvtxRangePushA("Device Memory Allocation");
    float4* d_verts; float* d_mat;
    CUDA_CHECK(cudaMalloc(&d_verts, vertBytes));
    CUDA_CHECK(cudaMalloc(&d_mat, matBytes));
    nvtxRangePop();

    // Copy to device
    nvtxRangePushA("Host-to-Device Transfer");
    CUDA_CHECK(cudaMemcpy(d_verts, h_verts, vertBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mat, h_mat, matBytes, cudaMemcpyHostToDevice));
    nvtxRangePop();

    int blockSize = 256;
    int gridSize = (numVerts + blockSize - 1) / blockSize;

    // Warmup
    nvtxRangePushA("Warmup");
    for (int i = 0; i < WARMUP_ITERS; i++)
        vertexShaderKernel<<<gridSize, blockSize>>>(d_verts, d_mat, numVerts);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    // Benchmark
    nvtxRangePushA("Benchmark Iterations");
    float times[BENCH_ITERS];
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < BENCH_ITERS; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        vertexShaderKernel<<<gridSize, blockSize>>>(d_verts, d_mat, numVerts);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    nvtxRangePop();

    // Stats
    float sum = 0, minT = 1e9, maxT = 0;
    for (int i = 0; i < BENCH_ITERS; i++) {
        sum += times[i];
        if (times[i] < minT) minT = times[i];
        if (times[i] > maxT) maxT = times[i];
    }
    float avg = sum / BENCH_ITERS;
    float mverts = (numVerts / (avg / 1000.0f)) / 1e6f;

    g_results[g_numResults++] = (BenchResult){
        "vertex_shader", res.name, avg, minT, maxT, mverts, "MVerts/s"
    };

    // Cleanup
    nvtxRangePushA("Cleanup");
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_verts));
    CUDA_CHECK(cudaFree(d_mat));
    free(h_verts); free(h_mat);
    nvtxRangePop();

    nvtxRangePop(); // End Vertex Shader Benchmark
}

void benchPixelShader(Resolution res) {
    nvtxRangePushA("Pixel Shader Benchmark");

    int W = res.width, H = res.height;
    int pixels = W * H;
    size_t fbBytes = pixels * 3 * sizeof(float);

    nvtxRangePushA("Host Memory Allocation");
    float* h_normals = (float*)malloc(fbBytes);
    fillRandomNormals(h_normals, pixels * 3);
    nvtxRangePop();

    nvtxRangePushA("Device Memory Allocation");
    float *d_fb, *d_normals;
    CUDA_CHECK(cudaMalloc(&d_fb, fbBytes));
    CUDA_CHECK(cudaMalloc(&d_normals, fbBytes));
    nvtxRangePop();

    nvtxRangePushA("Host-to-Device Transfer");
    CUDA_CHECK(cudaMemcpy(d_normals, h_normals, fbBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_fb, 0, fbBytes));
    nvtxRangePop();

    dim3 block(16, 16);
    dim3 grid((W+15)/16, (H+15)/16);

    nvtxRangePushA("Warmup");
    for (int i = 0; i < WARMUP_ITERS; i++)
        pixelShaderKernel<<<grid, block>>>(d_fb, d_normals, 0.5f, 0.7f, 0.5f, W, H);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    nvtxRangePushA("Benchmark Iterations");
    float times[BENCH_ITERS];
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < BENCH_ITERS; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        pixelShaderKernel<<<grid, block>>>(d_fb, d_normals, 0.5f, 0.7f, 0.5f, W, H);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    nvtxRangePop();

    float sum = 0, minT = 1e9, maxT = 0;
    for (int i = 0; i < BENCH_ITERS; i++) {
        sum += times[i]; if (times[i] < minT) minT = times[i]; if (times[i] > maxT) maxT = times[i];
    }
    float avg = sum / BENCH_ITERS;
    float mpix = (pixels / (avg / 1000.0f)) / 1e6f;

    g_results[g_numResults++] = (BenchResult){
        "pixel_shader", res.name, avg, minT, maxT, mpix, "MPixels/s"
    };

    nvtxRangePushA("Cleanup");
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_fb)); CUDA_CHECK(cudaFree(d_normals));
    free(h_normals);
    nvtxRangePop();
    nvtxRangePop();
}

void benchPostFxBlur(Resolution res) {
    nvtxRangePushA("Post-FX Blur Benchmark");

    int W = res.width, H = res.height;
    size_t bytes = W * H * 3 * sizeof(float);

    nvtxRangePushA("Memory Setup");
    float* h_data = (float*)malloc(bytes);
    fillRandom(h_data, W * H * 3);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_data, bytes, cudaMemcpyHostToDevice));
    nvtxRangePop();

    dim3 block(16, 16);
    dim3 grid((W+15)/16, (H+15)/16);

    nvtxRangePushA("Warmup");
    for (int i = 0; i < WARMUP_ITERS; i++)
        postFxBlurKernel<<<grid, block>>>(d_in, d_out, W, H);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    nvtxRangePushA("Benchmark Iterations");
    float times[BENCH_ITERS];
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < BENCH_ITERS; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        postFxBlurKernel<<<grid, block>>>(d_in, d_out, W, H);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    nvtxRangePop();

    float sum = 0, minT = 1e9, maxT = 0;
    for (int i = 0; i < BENCH_ITERS; i++) {
        sum += times[i]; if (times[i] < minT) minT = times[i]; if (times[i] > maxT) maxT = times[i];
    }
    float avg = sum / BENCH_ITERS;
    float mpix = (W * H / (avg / 1000.0f)) / 1e6f;

    g_results[g_numResults++] = (BenchResult){
        "postfx_blur", res.name, avg, minT, maxT, mpix, "MPixels/s"
    };

    nvtxRangePushA("Cleanup");
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    free(h_data);
    nvtxRangePop();
    nvtxRangePop();
}

void benchTextureSample(Resolution res) {
    nvtxRangePushA("Texture Sampling Benchmark");

    int texW = 1024, texH = 1024;
    int numSamples = res.width * res.height;
    size_t texBytes = texW * texH * 3 * sizeof(float);
    size_t uvBytes = numSamples * 2 * sizeof(float);
    size_t outBytes = numSamples * 3 * sizeof(float);

    nvtxRangePushA("Memory Setup");
    float* h_tex = (float*)malloc(texBytes);
    float* h_uvs = (float*)malloc(uvBytes);
    fillRandom(h_tex, texW * texH * 3);
    fillRandom(h_uvs, numSamples * 2);

    float *d_tex, *d_uvs, *d_out;
    CUDA_CHECK(cudaMalloc(&d_tex, texBytes));
    CUDA_CHECK(cudaMalloc(&d_uvs, uvBytes));
    CUDA_CHECK(cudaMalloc(&d_out, outBytes));
    CUDA_CHECK(cudaMemcpy(d_tex, h_tex, texBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uvs, h_uvs, uvBytes, cudaMemcpyHostToDevice));
    nvtxRangePop();

    int blockSize = 256;
    int gridSize = (numSamples + blockSize - 1) / blockSize;

    nvtxRangePushA("Warmup");
    for (int i = 0; i < WARMUP_ITERS; i++)
        textureSampleKernel<<<gridSize, blockSize>>>(d_tex, d_out, d_uvs, texW, texH, numSamples);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    nvtxRangePushA("Benchmark Iterations");
    float times[BENCH_ITERS];
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < BENCH_ITERS; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        textureSampleKernel<<<gridSize, blockSize>>>(d_tex, d_out, d_uvs, texW, texH, numSamples);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    nvtxRangePop();

    float sum = 0, minT = 1e9, maxT = 0;
    for (int i = 0; i < BENCH_ITERS; i++) {
        sum += times[i]; if (times[i] < minT) minT = times[i]; if (times[i] > maxT) maxT = times[i];
    }
    float avg = sum / BENCH_ITERS;
    float msamples = (numSamples / (avg / 1000.0f)) / 1e6f;

    g_results[g_numResults++] = (BenchResult){
        "texture_sample", res.name, avg, minT, maxT, msamples, "MSamples/s"
    };

    nvtxRangePushA("Cleanup");
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_tex)); CUDA_CHECK(cudaFree(d_uvs)); CUDA_CHECK(cudaFree(d_out));
    free(h_tex); free(h_uvs);
    nvtxRangePop();
    nvtxRangePop();
}

// =============================================================================
// JSON Output
// =============================================================================
void printResultsJSON() {
    printf("{\n");

    // GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  \"gpu\": \"%s\",\n", prop.name);
    printf("  \"compute_capability\": \"%d.%d\",\n", prop.major, prop.minor);
    printf("  \"vram_mb\": %lu,\n", prop.totalGlobalMem / (1024*1024));
    printf("  \"sm_count\": %d,\n", prop.multiProcessorCount);
    printf("  \"clock_mhz\": %d,\n", prop.clockRate / 1000);
    printf("  \"memory_clock_mhz\": %d,\n", prop.memoryClockRate / 1000);
    printf("  \"memory_bus_bits\": %d,\n", prop.memoryBusWidth);
    printf("  \"warmup_iters\": %d,\n", WARMUP_ITERS);
    printf("  \"bench_iters\": %d,\n", BENCH_ITERS);

    printf("  \"results\": [\n");
    for (int i = 0; i < g_numResults; i++) {
        BenchResult* r = &g_results[i];
        printf("    {\"kernel\": \"%s\", \"resolution\": \"%s\", "
               "\"avg_ms\": %.4f, \"min_ms\": %.4f, \"max_ms\": %.4f, "
               "\"throughput\": %.2f, \"unit\": \"%s\"}%s\n",
               r->kernel_name, r->resolution,
               r->avg_ms, r->min_ms, r->max_ms,
               r->throughput, r->throughput_unit,
               (i < g_numResults - 1) ? "," : "");
    }
    printf("  ]\n");
    printf("}\n");
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
    srand((unsigned int)time(NULL));

    // Print GPU info to stderr (JSON goes to stdout)
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "========================================\n");
    fprintf(stderr, "  FrameForge CUDA Shader Benchmark\n");
    fprintf(stderr, "========================================\n");
    fprintf(stderr, "  GPU:    %s\n", prop.name);
    fprintf(stderr, "  VRAM:   %lu MB\n", prop.totalGlobalMem / (1024*1024));
    fprintf(stderr, "  SMs:    %d\n", prop.multiProcessorCount);
    fprintf(stderr, "  Clock:  %d MHz\n", prop.clockRate / 1000);
    fprintf(stderr, "========================================\n\n");

    nvtxRangePushA("FrameForge Full Benchmark Suite");

    for (int r = 0; r < NUM_RESOLUTIONS; r++) {
        Resolution res = RESOLUTIONS[r];
        fprintf(stderr, "  [%s] (%dx%d)\n", res.name, res.width, res.height);

        char rangeName[64];
        snprintf(rangeName, sizeof(rangeName), "Resolution: %s", res.name);
        nvtxRangePushA(rangeName);

        fprintf(stderr, "    Running vertex_shader...   ");
        benchVertexShader(res);
        fprintf(stderr, "%.2f ms avg\n", g_results[g_numResults-1].avg_ms);

        fprintf(stderr, "    Running pixel_shader...    ");
        benchPixelShader(res);
        fprintf(stderr, "%.2f ms avg\n", g_results[g_numResults-1].avg_ms);

        fprintf(stderr, "    Running postfx_blur...     ");
        benchPostFxBlur(res);
        fprintf(stderr, "%.2f ms avg\n", g_results[g_numResults-1].avg_ms);

        fprintf(stderr, "    Running texture_sample...  ");
        benchTextureSample(res);
        fprintf(stderr, "%.2f ms avg\n", g_results[g_numResults-1].avg_ms);

        nvtxRangePop(); // Resolution
        fprintf(stderr, "\n");
    }

    nvtxRangePop(); // Full Benchmark Suite

    // Output JSON to stdout
    printResultsJSON();

    fprintf(stderr, "Done! %d benchmarks completed.\n", g_numResults);
    return 0;
}