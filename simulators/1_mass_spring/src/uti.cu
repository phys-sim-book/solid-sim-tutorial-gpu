#include "uti.h"
#include <muda/cub/device/device_reduce.h>
float devicesum(muda::DeviceBuffer<float> &buffer)
{
    float total_energy = 0.0f;         // Result of the reduction
    float *d_out;                      // Device memory to store the result of the reduction
    cudaMalloc(&d_out, sizeof(float)); // Allocate memory for the result

    // DeviceReduce is assumed to be part of the 'muda' library or similar
    muda::DeviceReduce().Sum(buffer.data(), d_out, buffer.size());

    // Copy the result back to the host
    cudaMemcpy(&total_energy, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_out);
    return total_energy;
}