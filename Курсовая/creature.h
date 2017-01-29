#ifndef CREATURE_HEADER
#define CREATURE_HEADER

#define SUBSTANCE_LENGTH 128

struct cell{
	unsigned int v[SUBSTANCE_LENGTH];
	int dv[SUBSTANCE_LENGTH];
};

//v[2], v[3], v[4] -- rgb

struct creature{
	int n;
	struct cell* cells;
};

#endif