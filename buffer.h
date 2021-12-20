#include <stdint.h>
#include <stdio.h>

// struct buffer_struct {
//     uint64_t *buffer_addrs;
//     int _max_size;
//     int _counter;
//     uint64_t last_ele;

//     buffer_struct() = default;
// };

// __device__ void print_single_mem_addr(uint64_t addr) {
//         printf(" 0x%016lx\n", addr);
// }

// __device__ void print_mem_addr_from_to(struct buffer_struct* buffer, int from, int to) {
//     for (int i = from; i < to; i++) {
//         print_single_mem_addr(buffer->buffer_addrs[i]);
//     }
// }


// __device__ void push_back(struct buffer_struct* buffer, uint64_t mem_addr) {
//     if (mem_addr == buffer->last_ele) {
//         return;
//     }
//     ++buffer->_counter;
//     if (buffer->_counter == buffer->_max_size) {
//         print_mem_addr_from_to(buffer, 0, buffer->_max_size);
//         buffer->_counter = 0;
//     }
//     buffer->_buffer_addrs[buffer->_counter] = mem_addr;
// }

#define BUFFER_SIZE 1000
class Buffer
{
private:
    
    int _counter;
    uint64_t last_ele;
    uint64_t _buffer_addrs[BUFFER_SIZE];
    uint64_t _repeat_time[BUFFER_SIZE];
    int _max_size;
    int _repeat_counter;

public:
    // uint64_t *_buffer_addrs;
    Buffer() {};

    void init() {
        _max_size = BUFFER_SIZE;
        printf("inited\n");
    }

    // Buffer(uint64_t* buffer_addrs, int max_size):_buffer_addrs(buffer_addrs),_max_size(max_size) {};
    __device__ __forceinline__ void print_single_mem_addr(uint64_t addr) {
        printf(" 0x%016lx\n", addr);
    }

    __device__ __forceinline__ void print_single_mem_addr_and_repeat(uint64_t addr,int repeat) {
        printf(" 0x%016lx  %d\n", addr, repeat);
    }

    __device__ __forceinline__ void print_mem_addr_from_to(int from, int to) {
        for (int i = from; i < to; i++) {
            print_single_mem_addr(_buffer_addrs[i]);
        }
    }

    __device__ __forceinline__ void print_mem_addr_and_repeat_from_to(int from, int to) {
        for (int i = from; i < to; i++) {
            print_single_mem_addr_and_repeat(_buffer_addrs[i],_repeat_time[i]);
        }
    }

    __device__ __forceinline__ void push_back(uint64_t mem_addr) {
        if (mem_addr == last_ele) {
            _repeat_counter++;
            return;
        }
        last_ele = mem_addr;
        _repeat_time[_counter] = _repeat_counter + 1;
        _repeat_counter = 0;
        ++_counter;
        // printf("%d\n",_counter);
        if (_counter == BUFFER_SIZE) {
            print_mem_addr_and_repeat_from_to(0, BUFFER_SIZE);
            _counter = 0;
        }
        _buffer_addrs[_counter] = mem_addr;
    }

    void host_print() {
        for (int i = 0; i < _counter; i++) {
            printf(" 0x%016lx  %d\n", _buffer_addrs[i], _repeat_time[i]);
        }
    }
};

