#ifndef KERNEL_H
#define KERNEL_H

#define MAX_THREAD_NUM 512
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t blurWithCuda(struct creature * creature, struct matrix * matrix);
cudaError_t calcWithCuda(struct creature *creature, struct genome* genome);

#endif