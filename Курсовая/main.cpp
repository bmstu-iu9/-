#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "genome.h"
#include "creature.h"
#include "bmp_writer.h"
#include "main.h"
#include "kernel.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

void init_blur_matrix(struct matrix ** matrix){
	*matrix = (struct matrix*)calloc(1, sizeof(struct matrix));
	(*matrix)->size = 3; 
	(*matrix)->val = (float*)calloc((*matrix)->size * (*matrix)->size, sizeof(float));
	(*matrix)->val[0] = 1;
	(*matrix)->val[1] = 2;
	(*matrix)->val[2] = 1;
	(*matrix)->val[3] = 2;
	(*matrix)->val[4] = 4;
	(*matrix)->val[5] = 2;
	(*matrix)->val[6] = 1;
	(*matrix)->val[7] = 2;
	(*matrix)->val[8] = 1;
	(*matrix)->norm_rate = 0;
	for(int i = 0; i < (*matrix)->size * (*matrix)->size; i++){
		(*matrix)->norm_rate += (*matrix)->val[i];
	}
}

int main(int argc, char **argv)
{
	struct genome *genome;
	struct creature *creature, *standard; 
	struct matrix * matrix;
	
	if(argc != 3){
		printf("Unsupported command. Type --help to get help\n");
		return -1;
	}
	if(strcmp(argv[1], "-i") == 0){
		genome = (struct genome*)malloc(sizeof(struct genome));
		load_genome(genome, argv[2]);
	}
	else if(strcmp(argv[1], "-rand") == 0){
		genome = (struct genome*)malloc(sizeof(struct genome));
		init_rand_genome(genome);
		save_genome(genome, argv[2]);
	}
	else{
		 printf("Unsupported command. Type --help to get help\n");
		 return -1;
	}
	
	init_creature(&creature);
	for(int i = 0; i < creature->n; i++){
		for(int j = 0; j < creature->n; j++){
			printf("%d %d %d\n", creature->cells[i * creature->n + j].dv[0], creature->cells[i * creature->n + j].dv[1], creature->cells[i * creature->n + j].dv[2]);
		}
	}
	/*creature = (struct creature*)malloc(sizeof(struct creature));
	creature->n = 4;
	creature->cells = (struct cell*)calloc(creature->n * creature->n, sizeof(struct cell));
	for(int i = 0; i < creature->n; i++){
		for(int j = 0; j < creature->n; j++){
			for(int k = 0; k < SUBSTANCE_LENGTH; k++){
				if(i * creature->n + j >= creature->n * creature->n / 2){
					creature->cells[j * creature->n + i].v[k] = 255;
				}
				else 
					creature->cells[j * creature->n + i].v[k] = 0;
			}
		}
	}
	for(int i = 0; i < creature->n; i++){
		for(int j = 0; j < creature->n; j++){
			printf("%d ", creature->cells[i * creature->n + j].v[0]);
		}
		printf("\n");
	}*/
	init_blur_matrix(&matrix);
	int step = 0;
	char path[FILENAME_MAX] = {0};
	cudaError_t cudaStatus = cudaSuccess;
	/*cudaStatus = blurWithCuda(creature, matrix);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "blurWithCuda failed!");
		}
	printf("after blur\n");
	for(int i = 0; i < creature->n; i++){
		for(int j = 0; j < creature->n; j++){
			printf("%d ", creature->cells[i * creature->n + j].dv[0]);
		}
		printf("\n");
	}*/
	/*puts("beore blur\n");
	for(int i = 0; i < creature->n; i++){
			for(int j = 0; j < creature->n; j++){
				printf("%d ", creature->cells[i * creature->n + j].v[0]);
			}
			printf("\n");
		}
		printf("\n");
	for(int i = 0; i < creature->n; i++){
		for(int j = 0; j < creature->n; j++){
			printf("%d ", creature->cells[i * creature->n + j].v[1]);
		}
		printf("\n");
	}*/
	while(creature->n < MAX_CREATURE_SIZE){
		cudaStatus = calcWithCuda(creature, genome);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "calcWithCuda failed!");
		}
		printf("creature size = %d\n", creature->n);
		apply_calc_changes(creature);
		cudaStatus = blurWithCuda(creature, matrix);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "blurWithCuda failed!");
		}
		apply_blur_changes(creature);
		/*printf("after blur\n");
		for(int i = 0; i < creature->n; i++){
			for(int j = 0; j < creature->n; j++){
				printf("%d ", creature->cells[i * creature->n + j].v[0]);
			}
			printf("\n");
		}
		printf("\n");
		for(int i = 0; i < creature->n; i++){
			for(int j = 0; j < creature->n; j++){
				printf("%d ", creature->cells[i * creature->n + j].v[1]);
			}
			printf("\n");
		}*/
		/*for(int i = 0; i < creature->n; i++){
			for(int j = 0; j < creature->n; j++){
				printf("%d ", creature->cells[i * creature->n + j].dv[0]);
			}
			printf("\n");
		}*/
		step++;
		if(step % GROW_SIZE == 0){
			if(creature->n > 2){
				sprintf(path, "output_image%d.bmp", step);
				create_img(creature, path);
			}			
			creature = grow(creature);
		}
	}
	
	create_img(creature, "output_final.bmp");
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
	//int sim = similarity(creature, standard);
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
	if(cudaMalloc((void**)d_cond, global_cond_length * sizeof(unsigned char)) != cudaSuccess)
		puts("ERROR: Unable to allocate cond-vector");
	if(cudaMalloc((void**)d_oper, global_oper_length * sizeof(unsigned char)) != cudaSuccess)
		puts("ERROR: Unable to allocate oper-vector");
	if(cudaMemcpy(*d_cond, cond, global_cond_length * sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess)
		puts("ERROR: Unable to copy cond-vector");
	if(cudaMemcpy(*d_oper, oper, global_oper_length * sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess)
		puts("ERROR: Unable to copy oper-vector");
	return;
}

void copy_after_kernel(struct creature *creature, unsigned int *v, int *dv){
	for(int i = 0; i < creature->n; i++){
		for(int j = 0; j < creature->n; j++){
			for(int k = 0; k < SUBSTANCE_LENGTH; k++){
				creature->cells[i * creature->n + j].v[k] = v[(i * creature->n + j) * (SUBSTANCE_LENGTH - 1) + k];
				creature->cells[i * creature->n + j].dv[k] = dv[(i * creature->n + j) * (SUBSTANCE_LENGTH - 1) + k];
			}
		//	printf("%d ", creature->cells[i * creature->n + j].dv[1]);
		}
	}
	//	printf("\n");
}

void init_arrays(unsigned int **v, int **dv, struct creature * creature){
	*v = NULL; 
	*v = (unsigned int*)calloc(creature->n * creature->n * SUBSTANCE_LENGTH, sizeof(unsigned int));
	*dv = NULL;
	*dv = (int*)calloc(creature->n * creature->n * SUBSTANCE_LENGTH, sizeof(int));
	for(int i = 0; i < creature->n; i++){
		for(int j = 0; j < creature->n; j++){
			for(int k = 0; k < SUBSTANCE_LENGTH; k++){
				(*v)[(i * creature->n + j) * (SUBSTANCE_LENGTH - 1) + k] = creature->cells[i * creature->n + j].v[k];
				(*dv)[(i * creature->n + j) * (SUBSTANCE_LENGTH - 1) + k] = creature->cells[i * creature->n + j].dv[k];
			}
			//printf("%d ", (*v)[(i * creature->n + j) * (SUBSTANCE_LENGTH - 1)]);
		}
	}
}