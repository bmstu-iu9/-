#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "genome.h"
#include "creature.h"
#include "bmp_writer.h"
#include "main.h"
#include "kernel.h"
#include <stdio.h>
#include <string.h>

#define FILE_NAME_LENGTH 20

void init_blur_matrix(struct matrix ** matrix){
	*matrix = (struct matrix*)calloc(1, sizeof(struct matrix));
	(*matrix)->size = 3; 
	(*matrix)->val = (float*)calloc((*matrix)->size * (*matrix)->size, sizeof(float));
	(*matrix)->val[0] = 1/16;
	(*matrix)->val[1] = 1/8;
	(*matrix)->val[2] = 1/16;
	(*matrix)->val[3] = 1/8;
	(*matrix)->val[4] = 1/4;
	(*matrix)->val[5] = 1/8;
	(*matrix)->val[6] = 1/16;
	(*matrix)->val[7] = 1/8;
	(*matrix)->val[8] = 1/16;	
}

int main(int argc, char **argv)
{
	struct genome *genome;
	struct creature *creature; 
	struct matrix * matrix;
	
	if(argc != 3){
		printf("Unsupported command. Type --help to get help\n");
		return -1;
	}
	if(strcmp(argv[1], "-i") == 0){
		genome = (struct genome*)malloc(sizeof(struct genome));
		printf("%s\n", argv[1]);
	}
	else if(strcmp(argv[1], "-rand") == 0){
		genome = (struct genome*)malloc(sizeof(struct genome));
		init_rand_genome(genome);
	}
	else{
		 printf("Unsupported command. Type --help to get help\n");
		 return -1;
	}
	
	init_creature(&creature);
	init_blur_matrix(&matrix);
	/*cudaError_t cudaStatus;
	cudaStatus = calcWithCuda(creature, genome);
	grow(&creature);
	for(int i = 0; i < creature->n; i++){
		for(int j = 0; j < creature->n; j++){
			//(*creature)->cells[i * N + j].v[0] = (*creature)->cells[i * N + j].dv[0] = 1;
			printf("%d %d %d\n", creature->cells[i * creature->n + j].v[2], creature->cells[i * creature->n + j].v[3], creature->cells[i * creature->n + j].v[4]);
		}
	}*/
	int step = 0;
	char path[FILE_NAME_LENGTH] = {0};
	cudaError_t cudaStatus;
	while(creature->n < MAX_CREATURE_SIZE){
		if(step != 0 && step % GROW_SIZE == 0){
			if(creature->n > 2){
				sprintf(path, "output_image%d.bmp", step);
				create_img(creature, path);
			}			
			grow(&creature);
		}

		cudaStatus = calcWithCuda(creature, genome);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "calcWithCuda failed!");
		}
		printf("creature size = %d\n", creature->n);

		cudaStatus = blurWithCuda(creature, matrix);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "blurWithCuda failed!");
		}
		step++;
	}
	
	/*for(i = 0; i < creature->n; i++){
		for(j = 0; j < creature->n; j++){
			printf("%d %d %d\n", creature->cells[i * creature->n + j].v[2], creature->cells[i * creature->n + j].v[3], creature->cells[i * creature->n + j].v[4]);
		}
	}*/
	

	printf("%d\n", genome->length);
	
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
	
	free(genome);
	free(creature->cells);
	free(creature);
	free(matrix);
	
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

void init_arrays(unsigned int **v, int **dv, struct creature * creature){
	*v = NULL; 
	*v = (unsigned int*)calloc(creature->n * creature->n * SUBSTANCE_LENGTH, sizeof(unsigned int));
	*dv = NULL;
	*dv = (int*)calloc(creature->n * creature->n * SUBSTANCE_LENGTH, sizeof(int));
	for(int i = 0; i < creature->n; i++){
		for(int j = 0; j < creature->n; j++){
			for(int k = 0; k < SUBSTANCE_LENGTH; k++){
				(*v)[(i * creature->n + j) * SUBSTANCE_LENGTH + k] = creature->cells[i * creature->n + j].v[k];
				(*dv)[(i * creature->n + j) * SUBSTANCE_LENGTH + k] = creature->cells[i * creature->n + j].dv[k];
			}		
		}
	}
}