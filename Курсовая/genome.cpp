#include "genome.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

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

void load_genome(struct genome * genome, const char * path){
	FILE *fp;
	if ((fp = fopen(path, "rb"))==NULL) {
		printf ("Cannot open genome file.\n");
		return;
	}
	genome->length = 0;
	if(fread(&(genome->length), 1, 1, fp) != 1)
		printf("error on reading!");
	printf("genome length = %d\n", genome->length);
	genome->genes = (struct gene*)calloc(genome->length, sizeof(struct gene));
	for(int i = 0; i < genome->length; i++){
		fread(&(genome->genes[i].cond_length), 1, 1, fp);
		printf("cond length = %d\n", genome->genes[i].cond_length);
		genome->genes[i].cond = (struct cond*)calloc(genome->genes[i].cond_length, sizeof(struct cond));
		for(int j = 0; j < genome->genes[i].cond_length; j++){
			unsigned char *substance, *sign, *threshold;
			substance = (unsigned char*)calloc(1, sizeof(unsigned char));
			sign = (unsigned char*)calloc(1, sizeof(unsigned char));
			threshold = (unsigned char*)calloc(1, sizeof(unsigned char));
			fread(substance, 1, 1, fp);
			genome->genes[i].cond[j].substance = *substance;
			printf("cond subst = %d\n", genome->genes[i].cond[j].substance);
			fread(sign, 1, 1, fp);
			genome->genes[i].cond[j].sign = *sign;
			printf("cond sign = %d\n", genome->genes[i].cond[j].sign);
			fread(threshold, 1, 1, fp);
			genome->genes[i].cond[j].threshold = * threshold;
			printf("cond threshold = %d\n", genome->genes[i].cond[j].threshold);
			free(substance);
			free(sign);
			free(threshold);
		}
		fread(&(genome->genes[i].oper_length), 1, 1, fp);
		printf("oper length = %d\n", genome->genes[i].oper_length);
		genome->genes[i].operons = (struct operon*)calloc(genome->genes[i].oper_length, sizeof(struct operon));
		for(int j = 0; j < genome->genes[i].oper_length; j++){
			unsigned char *substance, *sign, *rate;
			substance = (unsigned char*)calloc(1, sizeof(unsigned char));
			sign = (unsigned char*)calloc(1, sizeof(unsigned char));
			rate = (unsigned char*)calloc(1, sizeof(unsigned char));
			fread(substance, 1, 1, fp);
			genome->genes[i].operons[j].substance = *substance;
			printf("oper substance = %d\n", genome->genes[i].operons[j].substance);
			fread(sign, 1, 1, fp);
			genome->genes[i].operons[j].sign = *sign;
			printf("oper sign = %d\n", genome->genes[i].operons[j].sign);
			fread(rate, 1, 1, fp);
			genome->genes[i].operons[j].rate = *rate;
			printf("oper rate = %d\n", genome->genes[i].operons[j].rate);
			free(substance);
			free(sign);
			free(rate);
		}
		
	}
}
