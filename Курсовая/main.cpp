#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "genome.h"
#include "creature.h"
#include "bmp_writer.h"
#include "main.h"
#include "kernel.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
	
	if(argc != 3){
		printf("Unsupported command. Type --help to get help\n");
		return -1;
	}
	if(strcmp(argv[1], "-i") == 0){
		printf("%s\n", argv[1]);
	}
	else if(strcmp(argv[1], "-rand") == 0){
		printf("%s\n", argv[1]);
	}
	else{
		 printf("Unsupported command. Type --help to get help\n");
		 return -1;
	}
	struct genome *genome; 
	struct creature *creature, *standard, *copy; 
	struct matrix * matrix;
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
			//printf("%d %d %d\n", creature->cells[i * creature->n + j].v[2], creature->cells[i * creature->n + j].v[3], creature->cells[i * creature->n + j].v[4]);
			//printf("%d %d\n", creature->cells[i * creature->n + j].v[0], creature->cells[i * creature->n + j].dv[0]);
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
	//cudaError_t cudaStatus = cudaSuccess;
	cudaError_t cudaStatus = calcWithCuda(creature, genome);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calcWithCuda failed!");
		goto Error;
	}
	printf("creature size = %d\n", creature->n);
	matrix->size = 2; //создание матрицы свертки. вынести в функцию
	matrix->val = (int*)calloc(matrix->size * matrix->size, sizeof(int));
	matrix->val[0] = 1;
	//cudaStatus = blurWithCuda(creature, matrix);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "blurWithCuda failed!");
		goto Error;
	}
	/*for(i = 0; i < creature->n; i++){
		for(j = 0; j < creature->n; j++){
			printf("%d %d %d\n", creature->cells[i * creature->n + j].v[2], creature->cells[i * creature->n + j].v[3], creature->cells[i * creature->n + j].v[4]);
		}
	}*/
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


void init_dev_creature(unsigned int *v, unsigned int **d_v, int *dv, int **d_dv, struct creature* creature ){
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
			creature->cells[i * creature->n + j].v[k] = v[(i * creature->n + j) * SUBSTANCE_LENGTH + k];
			creature->cells[i * creature->n + j].dv[k] = dv[(i * creature->n + j) * SUBSTANCE_LENGTH + k];
			}		
		}
	}
}