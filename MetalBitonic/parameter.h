//
//  parameter.h
//  MetalTest
//
//  Created by Giles Payne on 2024/11/21.
//

#pragma once

#ifndef _UINT32_T
#define _UINT32_T
typedef unsigned int uint32_t;
#endif /* _UINT32_T */

typedef enum {
    eLocalBitonicMergeSort = 0,
    eLocalDisperse,
    eBigFlip,
    eBigDisperse
} eAlgorithmVariant;


struct Parameters {
    uint32_t h;
    eAlgorithmVariant algorithm;
};
