#include "creature.h"
#include "genome.h"
#include "kernel.h"
#include "main.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

__device__ int calc_sigma(int x){
	return 1 / (1 + exp((double)(-x)));
}

__device__ unsigned char get_oper_rate(int num_gene, int num_oper, int* oper_offset, unsigned char* oper){
	return oper[oper_offset[num_gene] + 2 * num_oper] & 0x7f;
}

__device__ unsigned char get_oper_sign(int num_gene, int num_oper, int* oper_offset, unsigned char* oper){
	return oper[oper_offset[num_gene] + 2 * num_oper] & 0x80;
}

__device__ unsigned char get_oper_substnace(int num_gene, int num_oper, int* oper_offset, unsigned char* oper){
	return oper[oper_offset[num_gene] + 2 * num_oper + 1] & 0x7f;
}

__device__ unsigned char get_cond_threshold(int num_gene, int num_cond, int* cond_offset, unsigned char* cond){
	return cond[cond_offset[num_gene] + 2 * num_cond] & 0x7f;
}

__device__ unsigned char get_cond_sign(int num_gene, int num_cond, int* cond_offset, unsigned char* cond){
	return cond[cond_offset[num_gene] + 2 * num_cond] & 0x80;
}

__device__ unsigned char get_cond_substance(int num_gene, int num_cond, int* cond_offset, unsigned char* cond){
	return cond[cond_offset[num_gene] + 2 * num_cond + 1] & 0x7f;
}

//get i,j cell ,k value in vector -- (i * size + j) * SUBSTANCE_LENGTH + k

__global__ void calcKernel(int *dv, int creature_size, unsigned char* oper, int *oper_length, int* oper_offset, unsigned char* cond, int* cond_length, int* cond_offset, int genome_size)
{
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int x = blockDim.x*blockIdx.x + threadIdx.x;
    if (x >= creature_size || y >= creature_size)
		return;
	int k, l, p;
	int cur_cell = y * creature_size + x;
	for (k = 0; k < genome_size; k++){
		int *delta = (int*)malloc(cond_length[k] * sizeof(int));
		for (l = 0; l < cond_length[k]; l++){
			unsigned char cur_cond_sign = get_cond_sign(k, l, cond_offset, cond);
			unsigned char cur_cond_threshold = get_cond_threshold(k, l, cond_offset, cond);
			unsigned char cur_cond_substance = get_cond_substance(k, l, cond_offset, cond);
			/*printf("cur_cond_sign = %d\n", cur_cond_sign);
			printf("cur_cond_substance = %d\n", cur_cond_substance);
			printf("cur_cond_threshold = %d\n", cur_cond_threshold);*/
			delta[l] = cur_cond_sign
				? dv[cur_cell * SUBSTANCE_LENGTH + cur_cond_substance] - cur_cond_threshold
				: cur_cond_threshold - dv[cur_cell * SUBSTANCE_LENGTH + cur_cond_substance];
		}
		for (l = 0; l < oper_length[k]; l++){
			for (p = 0; p < cond_length[l]; p++){
				unsigned char cur_oper_substance = get_oper_substnace(k, l, oper_offset, oper);
				unsigned char cur_oper_rate = get_oper_rate(k, l, oper_offset, oper);
				unsigned char cur_oper_sign = get_oper_sign(k, l, oper_offset, oper);
				/*printf("cur_oper_sign = %d\n", cur_oper_sign);
				printf("cur_oper_substance = %d\n", cur_oper_substance);
				printf("cur_oper_rate = %d\n", cur_oper_rate);*/
				cur_oper_sign ? dv[cur_cell * SUBSTANCE_LENGTH + cur_oper_substance] -= (int)(cur_oper_rate * calc_sigma(delta[p])) :
					dv[cur_cell * SUBSTANCE_LENGTH + cur_oper_substance] += (int)(cur_oper_rate * calc_sigma(delta[p]));
			}
		}
		free(delta);
	}
}

__global__ void blurKernel(unsigned int *cr_v, unsigned int *cp_v, int c_size, float *m_val, int m_size){
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	if (x >= c_size || y >= c_size)
		return;
	int sz = m_size / 2;
	int core_point = x * c_size + y;
	int cur_cell_i = x;
	int cur_cell_j = y;
	int accum[SUBSTANCE_LENGTH] = { 0 };
	int k, l, p;
	for (k = 0; k < m_size; k++){
		for (l = 0; l < m_size; l++){
			cur_cell_i = x - sz + k;
			cur_cell_j = y - sz + l;
			if (cur_cell_i > 0 && cur_cell_i < c_size && cur_cell_j > 0 && cur_cell_j < c_size){
				for (p = 0; p < SUBSTANCE_LENGTH; p++){
					accum[p] += (int)(cp_v[((x * c_size + y) * SUBSTANCE_LENGTH + p)] * m_val[k * m_size + l]);
				}
			}
		}
	}
	for (p = 0; p < SUBSTANCE_LENGTH; p++){
		cr_v[core_point + p] = accum[p];
	}
}

cudaError_t calcWithCuda(struct creature *creature, struct genome* genome)
{
	cudaError_t cudaStatus;	
	int i, j;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!\n");
	}
	unsigned int *v, *d_v;
	int *dv, *d_dv;
	init_arrays(&v, &dv, creature);
	init_dev_creature(v, &d_v, dv, &d_dv, creature);
	unsigned char *cond, *oper, *d_cond, *d_oper;
	int global_cond_length = 0, global_oper_length = 0;
	for(i = 0; i < genome->length; i++){
		global_cond_length += genome->genes[i].cond_length;
		global_oper_length += genome->genes[i].oper_length;
	}
	cond = (unsigned char*)calloc(2 * global_cond_length, sizeof(unsigned char));
	oper = (unsigned char*)calloc(2 * global_oper_length, sizeof(unsigned char));
	int cur_offset = 0;
	for(i = 0; i < genome->length; i++){
		struct gene cur_gene = genome->genes[i];
		for(j = 0; j < 2 * cur_gene.cond_length; j+=2){
			cond[cur_offset + j] += cur_gene.cond[j].threshold;
			cond[cur_offset + j] += (cur_gene.cond[j].sign << 7);
			cond[cur_offset + j + 1] += cur_gene.cond[j].substance;
		}
		cur_offset += cur_gene.cond_length + 1;
	}

	cur_offset = 0;
	for(i = 0; i < genome->length; i++){
		struct gene cur_gene = genome->genes[i];
		for(j = 0; j < 2 * cur_gene.oper_length; j+=2){
			oper[cur_offset + j] += cur_gene.operons[j].rate;
			oper[cur_offset + j] += (cur_gene.operons[j].sign << 7);
			oper[cur_offset + j + 1] += cur_gene.operons[j].substance; //последний бит байта 0
		}
		cur_offset += cur_gene.oper_length + 1;
	}
	
	init_dev_genome(cond, &d_cond, oper, &d_oper, global_cond_length, global_oper_length);
	int *gen_cond_length, *gen_oper_length, *d_gen_cond_length = NULL, *d_gen_oper_length = NULL;
	int *gen_cond_offset, *gen_oper_offset, *d_gen_oper_offset, *d_gen_cond_offset;
	gen_cond_length = (int*)calloc(genome->length, sizeof(int));
	gen_oper_length = (int*)calloc(genome->length, sizeof(int));
	gen_oper_offset = (int*)calloc(genome->length, sizeof(int));
	gen_cond_offset = (int*)calloc(genome->length, sizeof(int));
	int tc_offset = 0, to_offset = 0;
	for(i = 0; i < genome->length; i++){
		gen_cond_offset[i] = tc_offset;
		gen_cond_offset[i] = to_offset;
		gen_cond_length[i] = genome->genes[i].cond_length;
		gen_oper_length[i] = genome->genes[i].oper_length;
		tc_offset += genome->genes[i].cond_length * 2;
		to_offset += genome->genes[i].oper_length * 2;
	}
	if(cudaMalloc((void**)&d_gen_cond_length, genome->length * sizeof(int)) != cudaSuccess)
		puts("ERROR: Unable to allocate cond-length vector");
	if(cudaMalloc((void**)&d_gen_oper_length, genome->length * sizeof(int)) != cudaSuccess)
		puts("ERROR: Unable to allocate oper-length vector");
	if(cudaMemcpy(d_gen_cond_length, gen_cond_length, genome->length * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		puts("ERROR: Unable to copy cond-length vector");
	if(cudaMemcpy(d_gen_oper_length, gen_oper_length, genome->length * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		puts("ERROR: Unable to copy oper-length vector");
	
	if(cudaMalloc((void**)&d_gen_cond_offset, genome->length * sizeof(int)) != cudaSuccess)
		puts("ERROR: Unable to allocate cond-offset vector");
	if(cudaMalloc((void**)&d_gen_oper_offset, genome->length * sizeof(int)) != cudaSuccess)
		puts("ERROR: Unable to allocate oper-offset vector");
	if(cudaMemcpy(d_gen_cond_offset, gen_cond_offset, genome->length * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		puts("ERROR: Unable to copy cond-offset vector");
	if(cudaMemcpy(d_gen_oper_offset, gen_oper_offset, genome->length * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		puts("ERROR: Unable to copy oper-offset vector");
	int threadNum = MAX_THREAD_NUM;
	dim3 blockSize = dim3(threadNum, 1, 1);
	dim3 gridSize = dim3(creature->n/threadNum + 1, creature->n, 1);
	calcKernel << <gridSize,blockSize>> >(d_dv, creature->n, d_oper, d_gen_oper_length, d_gen_oper_offset, d_cond, d_gen_cond_length, d_gen_cond_offset, genome->length);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calcKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcKernel\n", cudaStatus);
	}
	if(cudaMemcpy(v, d_v, creature->n * creature->n * SUBSTANCE_LENGTH * sizeof(unsigned int*), cudaMemcpyDeviceToHost) != cudaSuccess) 
		puts("ERROR: Unable to get v-vector from device\n");
	if(cudaMemcpy(dv, d_dv, creature->n * creature->n * SUBSTANCE_LENGTH *sizeof(int*), cudaMemcpyDeviceToHost) != cudaSuccess)
		puts("ERROR: Unable to get dv-vector from device\n");
	//puts("After calc kernel\n");
	/*for(i = 0; i < creature->n; i++){
		for(j = 0; j < creature->n; j++){
			printf("%d %d\n", v[i * creature->n + j], dv[i * creature->n + j]);
		}
	}*/
	copy_after_kernel(creature, v, dv);
	
	free(v);
	free(dv);
	free(cond);
	free(oper);
	free(gen_cond_length);
	free(gen_oper_length);
	cudaFree(d_v);
	cudaFree(d_dv);
	cudaFree(d_cond);
	cudaFree(d_oper);
	cudaFree(d_gen_cond_length);
	cudaFree(d_gen_oper_length);
	
	return cudaStatus;
}

cudaError_t blurWithCuda(struct creature * creature, struct matrix * matrix){
	cudaError_t cudaStatus;	
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!\n");
	}
	unsigned int *v, *cr_v, *cp_v;
	int *dv, *cr_dv, *cp_dv;
	float *m, *d_m;
	init_arrays(&v, &dv, creature);
	m = (float*)calloc(matrix->size * matrix->size, sizeof(float));
	memcpy(m, matrix->val, matrix->size * matrix->size * sizeof(float));
	
	init_dev_creature(v, &cr_v, dv, &cr_dv, creature);
	init_dev_creature(v, &cp_v, dv, &cp_dv, creature);

	if(cudaMalloc((void**)&d_m, matrix->size * matrix->size * sizeof(float)) != cudaSuccess){
		puts("ERROR: Unable to allocate convolution matrix\n");
	}
	if(cudaMemcpy(d_m, m,  matrix->size * matrix->size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess){
		puts("ERROR: Unable to copy matrix\n");
	}
	
	int threadNum = MAX_THREAD_NUM;
	dim3 blockSize = dim3(threadNum, 1, 1);
	dim3 gridSize = dim3(creature->n/threadNum + 1, creature->n, 1);
	blurKernel << <blockSize, gridSize>> >(cr_v, cp_v, creature->n, d_m, matrix->size);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calcKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcKernel!\n", cudaStatus);
	}
	if(cudaMemcpy(v, cr_v, creature->n * creature->n * SUBSTANCE_LENGTH * sizeof(unsigned int*), cudaMemcpyDeviceToHost) != cudaSuccess)
		puts("ERROR: Unable to get v-vector from device\n");
	if(cudaMemcpy(dv, cr_dv, creature->n * creature->n * SUBSTANCE_LENGTH *sizeof(int*), cudaMemcpyDeviceToHost) != cudaSuccess)
		puts("ERROR: Unable to get dv-vector from device\n");
	copy_after_kernel(creature, v, dv);
	
	free(v);
	free(dv);
	free(m);
	cudaFree(cr_v);
	cudaFree(cp_v);
	cudaFree(cr_dv);
	cudaFree(cp_dv);
	cudaFree(d_m);
	return cudaStatus;
}


