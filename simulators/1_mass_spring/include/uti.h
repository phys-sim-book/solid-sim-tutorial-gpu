#pragma once
#include <muda/muda.h>
#include <muda/container.h>

float devicesum(muda::DeviceBuffer<float> &buffer);
