#ifndef MAIN_H
#define MAIN_H
#include "creature.h"

#define SUBSTANCE_LENGTH 128
#define N 2
#define GROW_SIZE 10

void init_dev_creature(unsigned int *v, unsigned int **d_v, int *dv, int **d_dv, struct creature * creature);
void init_dev_genome(unsigned char *cond, unsigned char **d_cond, unsigned char *oper, unsigned char **d_oper, int global_cond_length, int global_oper_length);
void copy_after_kernel(struct creature *creature, unsigned int *v, int *dv);
void init_arrays(unsigned int **v, int **dv, struct creature * creature);

struct matrix{
	int size;
	float *val;
};

#endif