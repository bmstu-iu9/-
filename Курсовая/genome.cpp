#include "genome.h"
#include <stdlib.h>
#include <time.h>

void init_rand_genome(struct genome * genome){
	srand(time(NULL));
	genome->length = rand() % MAX_GENOME_SIZE;
	genome->genes = (struct gene*)calloc(genome->length, sizeof(struct gene));
	for(int i = 0; i < genome->length; i++){
		srand(i * genome->length);
		genome->genes[i].cond_length = rand() % MAX_COND_LENGTH;
		genome->genes[i].cond = (struct cond*)calloc(genome->genes[i].cond_length, sizeof(struct cond));
		for(int j = 0; j < genome->genes[i].cond_length; j++){
			srand(j * genome->length);
			genome->genes[i].cond[j].threshold = rand() % MAX_COND_VALUE;
			srand(j * genome->length + 1);
			genome->genes[i].cond[j].substance = rand() % MAX_COND_VALUE;
			srand(j * genome->length + 2);
			genome->genes[i].cond[j].sign = rand() % 2;
		}
		srand(i * 2 * genome->length);
		genome->genes[i].oper_length = rand() % MAX_OPERON_LENGTH;
		genome->genes[i].operons = (struct operon*)calloc(genome->genes[i].oper_length, sizeof(struct operon));
		for(int j = 0; j < genome->genes[i].oper_length; j++){
			srand(j * 2 * genome->length);
			genome->genes[i].operons[j].rate = rand() % MAX_OPERON_VALUE;
			srand(j * 2 * genome->length + 1);
			genome->genes[i].operons[j].substance = rand() % MAX_OPERON_VALUE;
			srand(j * 2 * genome->length + 2);
			genome->genes[i].operons[j].sign = rand() % 2;
		}
	}
}
