#include <iostream>
#include <chrono>
#include <vector>
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <regex>
#include <string>

#define M 4096
#define K 4096
#define N (2 * 4096)
#define BM 128
#define BN 128
#define BK 16
#define TM (BM / BK)
#define TN (BN / BK)
#define num_threads (BM * BN / (TM * TN))
#define NUM_TILEA (BM * BK / num_threads)
#define NUM_TILEB (BN * BK / num_threads)

std::string loopUnrolling(const std::string& code, int threshold = 32) {
  // This regex pattern matches a for loop with the following structure:
  // for (int <varName> = <start>; <varName> < <end>; <varName>++) { <loopBody> }
  std::regex forLoopPattern(R"(for\s*\(\s*int\s+(\w+)\s*=\s*(\d+)\s*;\s*\1\s*<\s*(\d+)\s*;\s*\1\+\+\s*\)\s*\{\s*([^{}]*)\})");
  // Explanation of the regex:
  // for\s*\(                        : Matches 'for (' with optional whitespace
  // \s*int\s+                       : Matches 'int ' with optional whitespace
  // (\w+)                           : Captures the variable name (alphanumeric characters and underscores)
  // \s*=\s*                         : Matches '= ' with optional whitespace
  // (\d+)                           : Captures the start value (one or more digits)
  // \s*;\s*                         : Matches ';' with optional whitespace
  // \1\s*<\s*                       : Matches the captured variable name followed by '<' with optional whitespace
  // (\d+)                           : Captures the end value (one or more digits)
  // \s*;\s*                         : Matches ';' with optional whitespace
  // \1\+\+\s*                       : Matches the captured variable name followed by '++' with optional whitespace
  // \)\s*\{\s*                      : Matches ')' followed by '{' with optional whitespace
  // ([^{}]*)                        : Captures the loop body (anything except '{' or '}')
  // \}                              : Matches the closing '}'

  std::smatch match;
  std::string unrolledCode = code;
  while (std::regex_search(unrolledCode, match, forLoopPattern)) {
    std::string varName = match[1];
    int start = std::stoi(match[2]);
    int end = std::stoi(match[3]);
    std::string loopBody = match[4];

    if (end - start > threshold) {
      std::string skippedLoop =
        "for (int " + varName + " = " + std::to_string(start) + "; " +
        varName + " < " + std::to_string(end) + "; " +
        varName + "++) /* Skipped */ { " +
        loopBody + " }";
      unrolledCode = unrolledCode.substr(0, match.position()) + skippedLoop + unrolledCode.substr(match.position() + match.length());
    } else {
      std::string unrolledLoop;
      for (int i = start; i < end; ++i) {
        std::string unrolledIteration = loopBody;
        std::regex varPattern(varName);
        unrolledIteration = std::regex_replace(unrolledIteration, varPattern, std::to_string(i));
        unrolledLoop += unrolledIteration + "\n";
      }
      unrolledCode = unrolledCode.substr(0, match.position()) + unrolledLoop + unrolledCode.substr(match.position() + match.length());
    }
  }

  return unrolledCode;
}

const char* shaderSrc = R"(
// #include <metal_stdlib>
using namespace metal;

#define M 128
#define K 128
#define N (2 * 128)
#define BM 16
#define BN 16
#define BK 1
#define TM (BM / BK)
#define TN (BN / BK)
#define num_threads (BM * BN / (TM * TN))
#define NUM_TILEA (BM * BK / num_threads)
#define NUM_TILEB (BN * BK / num_threads)

kernel void matMulKernel(const device float* a [[buffer(0)]],
                         const device float* b [[buffer(1)]],
                         device float* c [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]]) {
    int threadRow = (lid / (BN / TN)) * TM;
    int threadCol = (lid % (BN / TN)) * TN;

    int aPtr = gid.y * BM * K;
    int bPtr = gid.x * BN * K;
    int cPtr = gid.y * BM * N + gid.x * BN;

    threadgroup float tileA[BM * BK];
    threadgroup float tileB[BK * BN];

    float threadResults[TM * TN] = {0};

    for (int bkidx = 0; bkidx < K; bkidx += BK) {
        for (int idx = 0; idx < NUM_TILEA; idx++) {
            tileA[lid + idx * num_threads] = a[aPtr + ((lid + idx * num_threads) / BK) * K + (lid + idx * num_threads) % BK];
        }
        for (int idx = 0; idx < NUM_TILEB; idx++) {
            tileB[lid + idx * num_threads] = b[bPtr + ((lid + idx * num_threads) / BK) * K + (lid + idx * num_threads) % BK];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            float localM[TM], localN[TN];
            for (int idx = 0; idx < TM; idx++) {
                localM[idx] = tileA[(threadRow + idx) * BK + dotIdx];
            }
            for (int idx = 0; idx < TN; idx++) {
                localN[idx] = tileB[(threadCol + idx) * BK + dotIdx];
            }
            for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
                for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
                    threadResults[resIdxM * TN + resIdxN] += localM[resIdxM] * localN[resIdxN];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
        for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
            c[cPtr + (threadRow + resIdxM) * N + (threadCol + resIdxN)] = threadResults[resIdxM * TN + resIdxN];
        }
    }
}
)";

//
//#import <Foundation/Foundation.h>
//#import <Metal/Metal.h>
//#import <QuartzCore/CAMetalLayer.h>


  void startCapture() {
    if (![[NSProcessInfo processInfo].environment[@"METAL_CAPTURE_ENABLED"] boolValue]) {
      NSLog(@"METAL_CAPTURE_ENABLED is not set. Please set it to 1 to enable Metal capture.");
      return;
    }
    
    MTLCaptureDescriptor *descriptor = [[MTLCaptureDescriptor alloc] init];
    descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
    descriptor.outputURL = [NSURL fileURLWithPath:@"gpu.cpp.gputrace"];

    NSFileManager *fileManager = [NSFileManager defaultManager];
    if ([fileManager fileExistsAtPath:@"gpu.cpp.gputrace"]) {
      NSError *error = nil;
      [fileManager removeItemAtPath:@"gpu.cpp.gputrace" error:&error];
      if (error) {
        NSLog(@"Error deleting existing gpu.cpp.gputrace directory: %@", error);
        return;
      } else {
        NSLog(@"Deleted existing gpu.cpp.gputrace directory.");
      }
    }

    NSError *error = nil;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      NSLog(@"MTLCreateSystemDefaultDevice returned nil. Metal may not be supported on this system.");
      return;
    }
    descriptor.captureObject = device;
    
    BOOL success = [MTLCaptureManager.sharedCaptureManager startCaptureWithDescriptor:descriptor error:&error];
    if (!success) {
        NSLog(@" error capturing mtl => %@ ", [error localizedDescription] );
    }
  }

  void stopCapture() {
    [MTLCaptureManager.sharedCaptureManager stopCapture];
  }

int main() {
    setenv("METAL_CAPTURE_ENABLED", "1", 1);
    @autoreleasepool {
        startCapture();
        // Create Metal device and command queue
        NSArray *devices = MTLCopyAllDevices();
        if (devices.count <= 0) {
            NSLog(@"No Metal devices found.");
            return -1;
        }
        id<MTLDevice> device = devices[0];
        std::cout << "Metal device successfully created: " << [[device name] UTF8String] << std::endl;
        if (!device) {
            std::cerr << "Error: Metal is not supported on this device." << std::endl;
            return -1;
        }
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            std::cerr << "Error: Failed to create command queue." << std::endl;
            return -1;
        }

        // Allocate memory
        std::vector<float> a(M * K, 1.0f); // Initialize with dummy data
        std::vector<float> b(K * N, 1.0f); // Initialize with dummy data
        std::vector<float> c(M * N, 0.0f);

        id<MTLBuffer> aBuffer = [device newBufferWithBytes:a.data() length:M * K * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuffer = [device newBufferWithBytes:b.data() length:K * N * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> cBuffer = [device newBufferWithBytes:c.data() length:M * N * sizeof(float) options:MTLResourceStorageModeShared];

        if (!aBuffer || !bBuffer || !cBuffer) {
            std::cerr << "Error: Failed to create buffers." << std::endl;
            return -1;
        }

        // Load Metal kernel function
        NSError *error = nullptr;
    std::string optShader = loopUnrolling(shaderSrc);
    std::cout << optShader;
        NSString *source = [NSString stringWithUTF8String:optShader.c_str()];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            std::cerr << "Error creating Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
            return -1;
        }
        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"matMulKernel"];
        if (!kernelFunction) {
            std::cerr << "Error creating Metal function." << std::endl;
            return -1;
        }
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (!pipelineState) {
            std::cerr << "Error creating compute pipeline state: " << [[error localizedDescription] UTF8String] << std::endl;
            return -1;
        }

        // Create Metal command buffer and compute command encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

        // Configure kernel execution
        MTLSize gridSize = MTLSizeMake((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
        MTLSize threadGroupSize = MTLSizeMake(BM * BN / (TM * TN), 1, 1);

        [computeEncoder setComputePipelineState:pipelineState];
        [computeEncoder setBuffer:aBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:bBuffer offset:0 atIndex:1];
        [computeEncoder setBuffer:cBuffer offset:0 atIndex:2];
        [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
        [computeEncoder endEncoding];

        // Record the start time
        auto start = std::chrono::high_resolution_clock::now();

        // Launch kernel
    int niter = 5;
        for (int i = 0; i < niter; i++) {
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            [computeEncoder setComputePipelineState:pipelineState];
            [computeEncoder setBuffer:aBuffer offset:0 atIndex:0];
            [computeEncoder setBuffer:bBuffer offset:0 atIndex:1];
            [computeEncoder setBuffer:cBuffer offset:0 atIndex:2];
            [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
            [computeEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }

        // Record the stop time
        auto stop = std::chrono::high_resolution_clock::now();

        // Calculate elapsed time
        std::chrono::duration<float, std::milli> elapsed = stop - start;
        float milliseconds = elapsed.count() / niter;

        // Calculate FLOPS
        float flops = 2.0f * M * N * K;
        float gflops = flops / (milliseconds * 1e6);

        std::cout << "Execution time / iterations: " << milliseconds << " ms\n";
        std::cout << "GFLOPS: " << gflops << "\n";

        // Copy result back to CPU
        memcpy(c.data(), [cBuffer contents], M * N * sizeof(float));

        // Check results
        // (Check code here)
        stopCapture();
        return 0;
    }
}
