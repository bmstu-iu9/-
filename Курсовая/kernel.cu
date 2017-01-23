#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "calcus.h"
#include "genome.h"
#include "creature.h"
#include "bmp_writer.h"
#include <stdio.h>
#include <string.h>

#define N 8
#define SUBSTANCE_LENGTH 128
#define MAX_THREAD_NUM 512

//убрать глобальные

struct genome *genome; 

struct creature *creature, *standard, *copy; 

//вынести в заголовочный
struct matrix{
	int size;
	int *val;
};


struct matrix * matrix;
//разбить на несколько файлов
cudaError_t calcWithCuda();
cudaError_t blurWithCuda();

__device__ unsigned char get_oper_rate(int num_gene, int num_oper, int* oper_offset, unsigned char* oper){
	return oper[oper_offset[num_gene] + 2 * num_oper] & 0x7f;
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

__global__ void calcKernel(unsigned int *v, int *dv, int creature_size, unsigned char* oper, int *oper_length, int* oper_offset, unsigned char* cond, int* cond_length, int* cond_offset, int genome_size)
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
			printf("cur_cond_threshold = %d\n", cur_cond_threshold);
			unsigned char cur_cond_substance = get_cond_substance(k, l, cond_offset, cond);
			delta[l] = cur_cond_sign
				? cur_cond_threshold - dv[cur_cell * SUBSTANCE_LENGTH + cur_cond_substance]
				: dv[cur_cell * SUBSTANCE_LENGTH + cur_cond_substance] - cur_cond_threshold;
		}
		for (l = 0; l < oper_length[k]; l++){
			for (p = 0; p < cond_length[l]; p++){
				unsigned char cur_oper_substance = get_oper_substnace(k, l, oper_offset, oper);
				unsigned char cur_oper_rate = get_oper_rate(k, l, oper_offset, oper);
				printf("cur_oper_rate = %d\n", cur_oper_rate);
				dv[cur_cell * SUBSTANCE_LENGTH + cur_oper_substance] +=
					(int)(cur_oper_rate * calc_sigma(delta[p]));
			}
		}
		free(delta);
	}
}

__global__ void blurKernel(unsigned int *cr_v, int *cr_dv, unsigned int *cp_v, int *cp_dv, int c_size, int *m_val, int m_size){
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int x = blockDim.x*blockIdx.x + threadIdx.x;
//check
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
					accum[p] += cr_v[((x * c_size + y) * SUBSTANCE_LENGTH + p)] * m_val[k * m_size + l];
				}
			}
		}
	}
	for (p = 0; p < SUBSTANCE_LENGTH; p++){
		cr_v[core_point + p] = accum[p];
	}
}

void copy_creature(struct creature *c, struct creature **new_c){
	int n = (*new_c)->n = c->n;
	(*new_c)->cells = (struct cell*)calloc(n * n, sizeof(struct cell));
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			for(int k = 0; k < SUBSTANCE_LENGTH; k++){
				(*new_c)->cells[i * n + j].v[k] = c->cells[i * n + j].v[k];
				(*new_c)->cells[i * n + j].dv[k] = c->cells[i * n + j].dv[k];
			}
		}
	}
}

int main()
{
	genome = (struct genome*)malloc(sizeof(struct genome));
	creature = (struct creature*)malloc(sizeof(struct creature));
	copy = (struct creature*)malloc(sizeof(struct creature));
	standard = (struct creature*)malloc(sizeof(struct creature));
	matrix = (struct matrix*)calloc(1, sizeof(struct matrix));
	creature->n = N;
	creature->cells = (struct cell*)calloc(creature->n * creature->n, sizeof(struct cell));
	int i, j;
	//init creature. вынести в функцию
	for(i = 0; i < creature->n; i++){
		for(j = 0; j < creature->n; j++){
			creature->cells[i * creature->n + j].v[0] = creature->cells[i * creature->n + j].dv[0] = 1;
			creature->cells[i * creature->n + j].v[2] = creature->cells[i * creature->n + j].v[3] = creature->cells[i * creature->n + j].v[4] = 128;
			printf("%d %d %d\n", creature->cells[i * creature->n + j].v[2], creature->cells[i * creature->n + j].v[3], creature->cells[i * creature->n + j].v[4]);
			printf("%d %d\n", creature->cells[i * creature->n + j].v[0], creature->cells[i * creature->n + j].dv[0]);
		}
	}
	//init genome. вынести в функцию
	genome->length = 4;
	genome->genes = (struct gene*)calloc(genome->length, sizeof(struct gene));
	for(i = 0; i < genome->length; i++){
		genome->genes[i].cond_length = 1;
		genome->genes[i].cond = (struct cond*)calloc(genome->genes[i].cond_length, sizeof(struct cond));
		for(j = 0; j < genome->genes[i].cond_length; j++){
			genome->genes[i].cond[j].threshold = 1;
		}
		genome->genes[i].oper_length = 1;
		genome->genes[i].operons = (struct operon*)calloc(genome->genes[i].oper_length, sizeof(struct operon));
		for(j = 0; j < genome->genes[i].oper_length; j++){
			genome->genes[i].operons[j].rate = 1;
		}
	}
	cudaError_t cudaStatus = cudaSuccess;
	//cudaError_t cudaStatus = calcWithCuda();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calcWithCuda failed!");
		goto Error;
	}
	printf("creature size = %d\n", creature->n);
	matrix->size = 2; //создание матрицы свертки. вынести в функцию
	matrix->val = (int*)calloc(matrix->size * matrix->size, sizeof(int));
	matrix->val[0] = 1;
	//cudaStatus = blurWithCuda();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "blurWithCuda failed!");
		goto Error;
	}
	for(i = 0; i < creature->n; i++){
		for(j = 0; j < creature->n; j++){
			printf("%d %d %d\n", creature->cells[i * creature->n + j].v[2], creature->cells[i * creature->n + j].v[3], creature->cells[i * creature->n + j].v[4]);
		}
	}
	create_img(creature); 

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		goto Error;
	}

Error:{
	free(genome);
	free(creature);
	free(standard);
	free(matrix);
}
	  return 0;
}


void init_dev_creature(unsigned int *v, unsigned int **d_v, int *dv, int **d_dv ){
	if(cudaMalloc((void**)d_v, creature->n * creature->n * SUBSTANCE_LENGTH * sizeof(unsigned int)) != cudaSuccess)
		puts("ERROR: Unable to allocate v-vector");
	if(cudaMalloc((void**)d_dv, creature->n * creature->n * SUBSTANCE_LENGTH * sizeof(int)) != cudaSuccess)
		puts("ERROR: Unable to allocate dv-vector");
	if(cudaMemcpy(*d_v, v, creature->n * creature->n * SUBSTANCE_LENGTH * sizeof(unsigned int), cudaMemcpyHostToDevice) != cudaSuccess)
		puts("ERROR: Unable to copy v-vector");
	if(cudaMemcpy(*d_dv, dv, creature->n * creature->n * SUBSTANCE_LENGTH * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		puts("ERROR: Unable to copy dv-vector");
	return;
}

void init_dev_genome(unsigned char *cond, unsigned char **d_cond, unsigned char *oper, unsigned char **d_oper, int global_cond_length, int global_oper_length){
	if(cudaMalloc((void**)d_cond, 2 * global_cond_length * sizeof(unsigned char)) != cudaSuccess)
		puts("ERROR: Unable to allocate cond-vector");
	if(cudaMalloc((void**)d_oper, 2 * global_oper_length * sizeof(unsigned char)) != cudaSuccess)
		puts("ERROR: Unable to allocate oper-vector");
	if(cudaMemcpy(*d_cond, cond, 2 * global_cond_length * sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess)
		puts("ERROR: Unable to copy cond-vector");
	if(cudaMemcpy(*d_oper, oper, 2 * global_oper_length * sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess)
		puts("ERROR: Unable to copy oper-vector");
	return;
}


void copy_after_kernel(struct creature *creature, unsigned int *v, int *dv){
	for(int i = 0; i < creature->n; i++){
		for(int j = 0; j < creature->n; j++){
			for(int k = 0; k < SUBSTANCE_LENGTH; k++){
			creature->cells[i * creature->n + j].v[k] = v[i * creature->n + j + k];
			creature->cells[i * creature->n + j].dv[k] = dv[i * creature->n + j + k];
			}		
		}
	}
}
//разбить на функции
cudaError_t calcWithCuda()
{
	cudaError_t cudaStatus;	
	int i, j, k;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!\n");
	}
	unsigned int *v = (unsigned int*)calloc(creature->n * creature->n * SUBSTANCE_LENGTH, sizeof(unsigned int));
	int *dv = (int*)calloc(creature->n * creature->n * SUBSTANCE_LENGTH, sizeof(int));
	for(i = 0; i < creature->n; i++){
		for(j = 0; j < creature->n; j++){
			for(k = 0; k < SUBSTANCE_LENGTH; k++){
				v[i * creature->n + j + k] = creature->cells[i * creature->n + j].v[k];
				dv[i * creature->n + j + k] = creature->cells[i * creature->n + j].dv[k];
			}		
		}
	}
	unsigned int *d_v;
	int *d_dv;
	init_dev_creature(v, &d_v, dv, &d_dv);
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
	for(j = 0; j < 2 * global_cond_length; j+=2){
		printf("cond[j] = %d\n", cond[j]);
	}
	cur_offset = 0;
	for(i = 0; i < genome->length; i++){
		struct gene cur_gene = genome->genes[i];
		for(j = 0; j < 2 * cur_gene.oper_length; j+=2){
			oper[cur_offset + j] += cur_gene.operons[j].rate; //последний бит байта 0
			oper[cur_offset + j + 1] += cur_gene.operons[j].substance; //последний бит байта 0
		}
		cur_offset += cur_gene.oper_length + 1;
	}
	
	for(j = 0; j < 2 * global_oper_length; j+=2){
		printf("oper[j] = %d\n", oper[j]);
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
	calcKernel << <gridSize,blockSize>> >(d_v, d_dv, creature->n, d_oper, d_gen_oper_length, d_gen_oper_offset, d_cond, d_gen_cond_length, d_gen_cond_offset, genome->length);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calcKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcKernel\n", cudaStatus);
	}
	if(cudaMemcpy(v, d_v, creature->n * creature->n * SUBSTANCE_LENGTH * sizeof(unsigned int*), cudaMemcpyDeviceToHost) != cudaSuccess) //копиаст. внести в функцию
		puts("ERROR: Unable to get v-vector from device\n");
	if(cudaMemcpy(dv, d_dv, creature->n * creature->n * SUBSTANCE_LENGTH *sizeof(int*), cudaMemcpyDeviceToHost) != cudaSuccess)
		puts("ERROR: Unable to get dv-vector from device\n");
	puts("After calc kernel\n");
	/*for(i = 0; i < creature->n; i++){
		for(j = 0; j < creature->n; j++){
			printf("%d %d\n", v[i * creature->n + j], dv[i * creature->n + j]);
		}
	}*/
	copy_after_kernel(creature, v, dv);
	
	free(v); //обработчик через goto
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

cudaError_t blurWithCuda(){
	cudaError_t cudaStatus;	
	int i, j, k;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!\n");
	}
	
	unsigned int *v, *cr_v, *cp_v;
	int *dv, *cr_dv, *cp_dv, *m, *d_m;
	
	v = (unsigned int*)calloc(creature->n * creature->n * SUBSTANCE_LENGTH, sizeof(unsigned int));
	dv = (int*)calloc(creature->n * creature->n * SUBSTANCE_LENGTH, sizeof(int));
	m = (int*)calloc(matrix->size * matrix->size, sizeof(int));
	memcpy(m, matrix->val, matrix->size * matrix->size * sizeof(int));
	for(i = 0; i < creature->n; i++){
		for(j = 0; j < creature->n; j++){
			for(k = 0; k < SUBSTANCE_LENGTH; k++){
				v[i * creature->n + j + k] = creature->cells[i * creature->n + j].v[k];
				dv[i * creature->n + j + k] = creature->cells[i * creature->n + j].dv[k];
			}		
		}
	}
	
	init_dev_creature(v, &cr_v, dv, &cr_dv);
	init_dev_creature(v, &cp_v, dv, &cp_dv);

	if(cudaMalloc((void**)&d_m, matrix->size * matrix->size * sizeof(int)) != cudaSuccess){
		puts("ERROR: Unable to allocate convolution matrix\n");
	}
	if(cudaMemcpy(d_m, m,  matrix->size * matrix->size * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess){
		puts("ERROR: Unable to copy matrix\n");
	}
	
	int threadNum = MAX_THREAD_NUM;
	dim3 blockSize = dim3(threadNum, 1, 1);
	dim3 gridSize = dim3(creature->n/threadNum + 1, creature->n, 1);
	blurKernel << <blockSize, gridSize>> >(cr_v, cr_dv, cp_v, cp_dv, creature->n, d_m, matrix->size);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calcKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcKernel!\n", cudaStatus);
	}
	if(cudaMemcpy(v, cr_v, creature->n * creature->n * SUBSTANCE_LENGTH * sizeof(unsigned int*), cudaMemcpyDeviceToHost) != cudaSuccess) //копипаст. внести в функцию
		puts("ERROR: Unable to get v-vector from device\n");
	if(cudaMemcpy(dv, cr_dv, creature->n * creature->n * SUBSTANCE_LENGTH *sizeof(int*), cudaMemcpyDeviceToHost) != cudaSuccess)
		puts("ERROR: Unable to get dv-vector from device\n");
	copy_after_kernel(creature, v, dv);
	
	free(v); //обработчик через goto
	free(dv);
	free(m);
	cudaFree(cr_v);
	cudaFree(cp_v);
	cudaFree(cr_dv);
	cudaFree(cp_dv);
	cudaFree(d_m);
	return cudaStatus;
}


