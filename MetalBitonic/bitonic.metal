//
//  bitonic.metal
//  MetalBitonic
//
//  Created by Giles Payne on 2024/11/17.
//

#include <metal_stdlib>
#include "parameter.h"
using namespace metal;

bool lesser_than(const int left, const int right){
    return left < right;
}

void global_compare_and_swap(device int* globalData,
                             int2 idx) {
    if (lesser_than(globalData[idx.x], globalData[idx.y])) {
        int tmp = globalData[idx.x];
        globalData[idx.x] = globalData[idx.y];
        globalData[idx.y] = tmp;
    }
}

void local_compare_and_swap(threadgroup int *localData,
                            int2 idx){
    if (lesser_than(localData[idx.x], localData[idx.y])) {
        int tmp = localData[idx.x];
        localData[idx.x] = localData[idx.y];
        localData[idx.y] = tmp;
    }
}

// Performs full-height flip (h height) over globally available indices.
void big_flip(device int *globalData,
              uint globalId,
              uint h) {
    uint t_prime = globalId;
    uint half_h = h >> 1; // Note: h >> 1 is equivalent to h / 2

    uint q       = ((2 * t_prime) / h) * h;
    uint x       = q     + (t_prime % half_h);
    uint y       = q + h - (t_prime % half_h) - 1;

    global_compare_and_swap(globalData, int2(x,y));
}

// Performs full-height disperse (h height) over globally available indices.
void big_disperse(device int *globalData,
                  uint globalId,
                  uint h) {
    uint t_prime = globalId;

    uint half_h = h >> 1; // Note: h >> 1 is equivalent to h / 2

    uint q       = ((2 * t_prime) / h) * h;
    uint x       = q + (t_prime % (half_h));
    uint y       = q + (t_prime % (half_h)) + half_h;

    global_compare_and_swap(globalData, int2(x,y));
}

// Performs full-height flip (h height) over locally available indices.
void local_flip(threadgroup int *localData,
                uint localId,
                uint h)
{
    uint t = localId;
    threadgroup_barrier(mem_flags::mem_none);

    uint half_h = h >> 1; // Note: h >> 1 is equivalent to h / 2
    int2 indices =
        int2( h * ( ( 2 * t ) / h ) ) +
        int2( t % half_h, h - 1 - ( t % half_h ) );

    local_compare_and_swap(localData, indices);
}

// Performs progressively diminishing disperse operations (starting with height h)
// on locally available indices: e.g. h==8 -> 8 : 4 : 2.
// One disperse operation for every time we can divide h by 2.
void local_disperse(threadgroup int *localData,
                    uint localId,
                    uint h)
{
    uint t = localId;
    for ( ; h > 1 ; h /= 2 ) {
        
        threadgroup_barrier(mem_flags::mem_none);

        uint half_h = h >> 1; // Note: h >> 1 is equivalent to h / 2
        int2 indices =
            int2( h * ( ( 2 * t ) / h ) ) +
            int2( t % half_h, half_h + ( t % half_h ) );

        local_compare_and_swap(localData, indices);
    }
}

void local_bms(threadgroup int *localData,
               uint localId,
               uint h){
    //uint t = localId;
    for (uint hh = 2; hh <= h; hh <<= 1) {  // note:  h <<= 1 is same as h *= 2
        local_flip(localData, localId, hh);
        local_disperse(localData, localId, hh/2);
    }
}

kernel void
bitonic(device int *globalData [[buffer(0)]],
        constant Parameters &parameters [[buffer(1)]],
        uint globalId [[thread_position_in_grid]],
        uint localId [[thread_position_in_threadgroup]],
        uint threadGroupSize [[threads_per_threadgroup]],
        uint threadGroupId [[threadgroup_position_in_grid]]) {

    threadgroup int localData[2048];

    uint t = localId;

    // We can use offset if we have more than one invocation.
    uint offset = threadGroupSize * 2 * threadGroupId;

    if (parameters.algorithm <= eLocalDisperse){
        // In case this shader executes a `local_` algorithm, we must
        // first populate the workgroup's local memory.
        localData[t*2]   = globalData[offset+t*2];
        localData[t*2+1] = globalData[offset+t*2+1];
    }

    switch (parameters.algorithm) {
        case eLocalBitonicMergeSort:
            local_bms(localData, localId, parameters.h);
        break;
        case eLocalDisperse:
            local_disperse(localData, localId, parameters.h);
        break;
        case eBigFlip:
            big_flip(globalData, globalId, parameters.h);
        break;
        case eBigDisperse:
            big_disperse(globalData, globalId, parameters.h);
        break;
    }

    // Write local memory back to buffer in case we pulled in the first place.
    if (parameters.algorithm <= eLocalDisperse) {
        threadgroup_barrier(mem_flags::mem_none);
        // push to global memory
        globalData[offset+t*2] = localData[t*2];
        globalData[offset+t*2+1] = localData[t*2+1];
    }
}



