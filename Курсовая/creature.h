#ifndef CREATURE_HEADER
#define CREATURE_HEADER

#define SUBSTANCE_LENGTH 128
#define MAX_CREATURE_SIZE 512
#define RED_COMPONENT 2
#define GREEN_COMPONENT 3
#define BLUE_COMPONENT 4

struct cell{
	unsigned int v[SUBSTANCE_LENGTH];
	int dv[SUBSTANCE_LENGTH];
};

//v[2], v[3], v[4] -- rgb

struct creature{
	int n;
	struct cell* cells;
};

void init_creature(struct creature ** creature);
struct creature* grow(struct creature * creature);
void apply_calc_changes(struct creature * creature);
void apply_blur_changes(struct creature * creature);
int similarity(struct creature * c, struct creature * e);

#endif