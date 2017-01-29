#include "creature.h"
#include "main.h"
#include <stdlib.h>

void init_creature(struct creature ** creature){
	*creature = (struct creature*)malloc(sizeof(struct creature));
	(*creature)->n = N;
	(*creature)->cells = (struct cell*)calloc(N * N, sizeof(struct cell));
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			(*creature)->cells[i * N + j].v[0] = (*creature)->cells[i * N + j].dv[0] = 1;
			(*creature)->cells[i * N + j].v[2] = (*creature)->cells[i * N + j].v[3] = (*creature)->cells[i * N + j].v[4] = 128;
		}
	}
}

void grow(struct creature ** creature){
	(*creature)->n = 2 * (*creature)->n;
	(*creature)->cells = (struct cell*)realloc((*creature)->cells, (*creature)->n * (*creature)->n * sizeof(struct cell));
}