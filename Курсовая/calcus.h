#include <math.h>

__device__ int calc_sigma(int x){
	return 1 / (1 + exp((double)(-x)));
}

