#include "genome.h"
#include <stdlib.h>
#include <stdint.h>
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

void save_genome(struct genome * genome, const char * path){
	FILE *fp;
	if ((fp = fopen(path, "wb"))==NULL) {
		printf ("Cannot open genome file.\n");
		return;
	}
	int i, j;
	for(i = 0; i < genome->length; i++){
		for(j = 0; j < genome->genes[i].cond_length; j++){
			uint16_t cond = 0;
			cond |= ((uint16_t)genome->genes[i].cond[j].sign) << 15;
			cond |= (uint16_t)genome->genes[i].cond[j].substance;
			cond |= ((uint16_t)genome->genes[i].cond[j].threshold) << 7;
			fwrite(&cond, sizeof(uint16_t), 1, fp);
			printf("cond = %x\n", cond);
		}
		for(j = 0; j < genome->genes[i].oper_length; j++){
			uint16_t oper = 0;
			oper |=	1 << 8;
			oper |= ((uint16_t)genome->genes[i].operons[j].sign) << 15;
			oper |= (uint16_t)genome->genes[i].operons[j].substance;
			oper |= ((uint16_t)genome->genes[i].operons[j].rate) << 7;
			fwrite(&oper, sizeof(uint16_t), 1, fp);
			printf("oper = %x\n", oper);
		}
	}
	fclose(fp);
}

bool get_flag(uint16_t val){
	return val & 0x80;
}

unsigned char get_substance(uint16_t val){
	return val & 0x7f;
}

unsigned char get_sign(uint16_t val){
	return (val & 0x8000) >> 15;
}

unsigned char get_threshold(uint16_t val){
	return (val & 0x7f00) >> 8;
}

unsigned char get_rate(uint16_t val){
	return (val & 0x7f00) >> 8;
}


void load_genome(struct genome * genome, const char * path){//todo
	FILE *fp;
	if ((fp = fopen(path, "rb"))==NULL) {
		printf ("Cannot open genome file.\n");
		return;
	}
	fseek(fp,0,SEEK_END);
    int size = (ftell(fp))/2;
	int i;
	rewind(fp);
	uint16_t* buffer = (uint16_t*)malloc(size * sizeof(uint16_t));
	if(fread(buffer, sizeof(uint16_t), size, fp) != size){
		printf("Error on reading genome!\n");
		return;
	}
	bool oper_flag = false;
	genome->length = 0;
	for(int i = 0; i < size; i++){
		if(oper_flag == false){
			oper_flag = get_flag(buffer[i]);
			if(oper_flag == true){
				genome->length += 1;
			}
		}
		else{
			oper_flag = get_flag(buffer[i]);
		}
	}
	genome->genes = (struct gene*)calloc(genome->length, sizeof(struct gene));
	int pos = 0;
	oper_flag = false;
	for(i = 0; i < size; i++){
		if(oper_flag == false){
			oper_flag = get_flag(buffer[i]);
			if(oper_flag == true){
				genome->genes[pos].oper_length++;
			}
			else{
				genome->genes[pos].cond_length++;
			}
		}
		else{
			oper_flag = get_flag(buffer[i]);
			if(oper_flag == true){
				genome->genes[pos].oper_length++;
			}
			else{
				pos++;
				genome->genes[pos].cond_length++;
			}
		}
	}
	for(i = 0; i < genome->length; i++){
		genome->genes[i].cond = (struct cond*)calloc(genome->genes[i].cond_length, sizeof(struct cond));
		genome->genes[i].operons = (struct operon*)calloc(genome->genes[i].oper_length, sizeof(struct operon));
	}
	pos = 0;
	oper_flag = false;
	int cur_cond = 0;
	int cur_oper = 0;
	for(i = 0; i < size; i++){
		if(oper_flag == false){
			oper_flag = get_flag(buffer[i]);
			if(oper_flag == true){
				genome->genes[pos].operons[cur_oper].substance = get_substance(buffer[i]);
				genome->genes[pos].operons[cur_oper].sign = get_sign(buffer[i]);
				genome->genes[pos].operons[cur_oper].rate = get_rate(buffer[i]);
				cur_oper++;
			}
			else{
				genome->genes[pos].cond[cur_cond].substance = get_substance(buffer[i]);
				genome->genes[pos].cond[cur_cond].sign = get_sign(buffer[i]);
				genome->genes[pos].cond[cur_cond].threshold = get_threshold(buffer[i]);
				cur_cond++;
			}
		}
		else{
			oper_flag = get_flag(buffer[i]);
			if(oper_flag == true){
				genome->genes[pos].operons[cur_oper].substance = get_substance(buffer[i]);
				genome->genes[pos].operons[cur_oper].sign = get_sign(buffer[i]);
				genome->genes[pos].operons[cur_oper].rate = get_rate(buffer[i]);
				cur_oper++;
			}
			else{
				pos++;
				cur_cond = cur_oper = 0;
				genome->genes[pos].cond[cur_cond].substance = get_substance(buffer[i]);
				genome->genes[pos].cond[cur_cond].sign = get_sign(buffer[i]);
				genome->genes[pos].cond[cur_cond].threshold = get_threshold(buffer[i]);
				cur_cond++;
			}
		}
	}
	for(i = 0; i < genome->length; i++){
		for(int j = 0; j < genome->genes[i].cond_length; j++){
			printf("cond = %d %d %d\n", genome->genes[i].cond[j].substance, genome->genes[i].cond[j].sign, genome->genes[i].cond[j].threshold);
		}
		for(int j = 0; j < genome->genes[i].oper_length; j++){
			printf("oper = %d %d %d\n", genome->genes[i].operons[j].substance, genome->genes[i].operons[j].sign, genome->genes[i].operons[j].rate);
		}
	}
	free(buffer);
	fclose(fp);
}
