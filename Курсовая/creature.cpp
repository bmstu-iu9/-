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
	(*creature)->cells[0].v[0] = 1;
	(*creature)->cells[1].v[0] = 1;
	(*creature)->cells[1].v[1] = 1; 
	(*creature)->cells[3].v[1] = 1;
}

void grow(struct creature ** creature){
	int old_size = (*creature)->n;
	int new_size = (*creature)->n = 2 * (*creature)->n;
	(*creature)->cells = (struct cell*)realloc((*creature)->cells, (*creature)->n * (*creature)->n * sizeof(struct cell));
	for(int i = 0; i < new_size; i++){
		for(int j = 0; j < new_size; j++){
			if((i * new_size + j) >= old_size * old_size){
				for(int k = 0; k < SUBSTANCE_LENGTH; k++){
					(*creature)->cells[i * new_size + j].v[k] = 0;
					(*creature)->cells[i * new_size + j].dv[k] = 0;
				}
			}
		}
	}
}