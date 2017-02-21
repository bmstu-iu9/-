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
	(*creature)->cells[0].v[0] = (*creature)->cells[0].dv[0] = 255;
	(*creature)->cells[1].v[0] = (*creature)->cells[1].dv[0] = 255;
	(*creature)->cells[2].v[0] = (*creature)->cells[2].dv[0] = 255;
	(*creature)->cells[3].v[0] = (*creature)->cells[3].dv[0] = 255;
	(*creature)->cells[4].v[0] = (*creature)->cells[4].dv[0] = 255;
	(*creature)->cells[5].v[0] = (*creature)->cells[5].dv[0] = 255;
	(*creature)->cells[6].v[0] = (*creature)->cells[6].dv[0] = 255;
	(*creature)->cells[7].v[0] = (*creature)->cells[7].dv[0] = 255;
	(*creature)->cells[3].v[1] = (*creature)->cells[3].dv[1] = 255; 
	(*creature)->cells[7].v[1] = (*creature)->cells[7].dv[1] = 255;
	(*creature)->cells[11].v[1] = (*creature)->cells[11].dv[1] = 255;
	(*creature)->cells[15].v[1] = (*creature)->cells[15].dv[1] = 255;
	(*creature)->cells[2].v[1] = (*creature)->cells[2].dv[1] = 255; 
	(*creature)->cells[6].v[1] = (*creature)->cells[6].dv[1] = 255;
	(*creature)->cells[10].v[1] = (*creature)->cells[10].dv[1] = 255;
	(*creature)->cells[14].v[1] = (*creature)->cells[14].dv[1] = 255;
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
	/*for(int i = 0; i < creature->n; i++){
		for(int j = 0; j < creature->n; j++){
			printf("index = %d %d ,new vals = %d %d\n", i ,j, creature->cells[i * creature->n + j].v[0], creature->cells[i * creature->n + j].v[1]);
		}
	}*/
	free(creature);
	return new_creature;
}

void apply_calc_changes(struct creature * creature){
	for(int i = 0; i < creature->n; i++){
		for(int j = 0; j < creature->n; j++){
			for(int k = 0; k < SUBSTANCE_LENGTH; k++){
				int temp = creature->cells[i * creature->n + j].v[k];
				if(temp + creature->cells[i * creature->n + j].dv[k] < 0){
					creature->cells[i * creature->n + j].v[k] = 0;
					continue;
				}
				else if(temp + creature->cells[i * creature->n + j].dv[k] > 255){
					creature->cells[i * creature->n + j].v[k] = 255;
					continue;
				}
				creature->cells[i * creature->n + j].v[k] += creature->cells[i * creature->n + j].dv[k];
			}
		}
	}
}

void apply_blur_changes(struct creature * creature){
	for(int i = 0; i < creature->n; i++){
		for(int j = 0; j < creature->n; j++){
			for(int k = 0; k < SUBSTANCE_LENGTH; k++){
				if(creature->cells[i * creature->n + j].dv[k] > 255){
					creature->cells[i * creature->n + j].v[k] = 255;
					continue;
				}
				else if(creature->cells[i * creature->n + j].dv[k] < 0){
					creature->cells[i * creature->n + j].v[k] = 0;
					continue;
				}
				creature->cells[i * creature->n + j].v[k] = creature->cells[i * creature->n + j].dv[k];
			}
		}
	}
}

int similarity(struct creature * c, struct creature * e){
    int red = 0, green = 0, blue = 0;
    if(c->n != e->n)
        return -1;
    for(int i = 0; i < c->n; i++){
        for(int j = 0; j < c->n; j++){
            red += abs((int)(c->cells[i * c->n + j].v[2] - e->cells[i * e->n + j].v[2]));
            green += abs((int)(c->cells[i * c->n + j].v[3] - e->cells[i * e->n + j].v[3]));
            blue += abs((int)(c->cells[i * c->n + j].v[4] - e->cells[i * e->n + j].v[4])); 
        }
    }
    return red + green + blue; 
}
