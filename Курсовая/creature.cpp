#include "creature.h"
#include "main.h"
#include <stdlib.h>
#include <stdio.h>

void init_creature(struct creature ** creature){
	*creature = (struct creature*)malloc(sizeof(struct creature));
	(*creature)->n = N;
	(*creature)->cells = (struct cell*)calloc(N * N, sizeof(struct cell));
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			for(int k = 0; k < SUBSTANCE_LENGTH; k++){
				(*creature)->cells[i * N + j].v[k] = (*creature)->cells[i * N + j].dv[k] = 0;
			}
		}
	}
	(*creature)->cells[0].v[0] = 127;
	(*creature)->cells[1].v[0] = 127;
	(*creature)->cells[1].v[1] = 127; 
	(*creature)->cells[3].v[1] = 127;
}

struct creature* grow(struct creature * creature){
	struct creature *new_creature = (struct creature*)malloc(sizeof(struct creature));
	int new_size = 2 * creature->n;
	new_creature->n = new_size;
	new_creature->cells = (struct cell*)calloc(new_creature->n * new_creature->n, sizeof(struct cell));
	for(int i = 0; i < new_size; i++){
		for(int j = 0; j < new_size; j++){
			for(int k = 0; k < SUBSTANCE_LENGTH; k++){
				new_creature->cells[i * new_size + j].v[k] = creature->cells[(i/2) * creature->n + j/2].v[k];
				new_creature->cells[i * new_size + j].dv[k] = creature->cells[(i/2) * creature->n + j/2].dv[k];
			}
		}
	}
	free(creature);
	return new_creature;
}