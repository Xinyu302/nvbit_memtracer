/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"
#include "buffer.h"

#include <vector>
#define PAGE_SIZE (1 << 12)
#define PAGE_ADDR(addr) (addr & ~(0xFFF))


// extern "C" __device__ __noinline__ void instrument_mem(int pred, int opcode_id,
//                                                        uint64_t addr, uint64_t buffer_ptr) {
    
//     /* all the active threads will compute the active mask */
//     const int active_mask = __ballot_sync(__activemask(), 1);

//     /* compute the predicate mask */
//     // const int predicate_mask = __ballot_sync(__activemask(), predicate);

//     /* each thread will get a lane id (get_lane_id is implemented in
//      * utils/utils.h) */
//     const int laneid = get_laneid();

//     /* get the id of the first active thread */
//     const int first_laneid = __ffs(active_mask) - 1;

//     /* count all the active thread */
//     // const int num_threads = __popc(predicate_mask);
    
//     /* if predicate is off return */
//     Buffer *buffer = (Buffer *)buffer_ptr;
//     if (!pred) {
//         return;
//     }

//     if (first_laneid == laneid) {
//         buffer->push_back(PAGE_ADDR(addr));
//     }
// }


extern "C" __device__ __noinline__ void instrument_mem(int pred, int opcode_id,
                                                       uint64_t addr, uint64_t buffer_ptr, uint64_t lock_ptr) {
    
    /* all the active threads will compute the active mask */
    const int active_mask = __ballot_sync(__activemask(), 1);

    /* compute the predicate mask */
    // const int predicate_mask = __ballot_sync(__activemask(), predicate);

    /* each thread will get a lane id (get_lane_id is implemented in
     * utils/utils.h) */
    const int laneid = get_laneid();

    /* get the id of the first active thread */
    const int first_laneid = __ffs(active_mask) - 1;

    /* count all the active thread */
    // const int num_threads = __popc(predicate_mask);
    
    /* if predicate is off return */
    Buffer *buffer = (Buffer *)buffer_ptr;
    if (!pred) {
        return;
    }

    if (first_laneid == laneid) {
        bool loopFlag = false;
        int *lock_var = (int * )lock_ptr;
        do {
            if ((loopFlag = atomicCAS(lock_var, 0, 1) == 0)) {
                buffer->push_back(PAGE_ADDR(addr));
                //Critical Section Code Here
            }
            __threadfence(); //Or __threadfence_block(), __threadfence_system() according to your Memory Fence demand
            if (loopFlag) atomicExch(lock_var, 0);
        } while (!loopFlag);
        // buffer->push_back(PAGE_ADDR(addr));

    }
}