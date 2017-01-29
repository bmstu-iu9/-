#ifndef BMP_WRITER_H
#define BMP_WRITER_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <malloc.h>
#include "creature.h"

#define _bitsperpixel 24
#define _planes 1
#define _compression 0
#define _pixelbytesize creature->n*creature->n*_bitsperpixel/8
#define _filesize _pixelbytesize+sizeof(bitmap)
#define _xpixelpermeter 0x130B //2835 , 72 DPI
#define _ypixelpermeter 0x130B //2835 , 72 DPI

#pragma pack(push,1)

typedef struct{
	uint8_t signature[2];
	uint32_t filesize;
	uint32_t reserved;
	uint32_t fileoffset_to_pixelarray;
} fileheader;

typedef struct{
	uint32_t dibheadersize;
	uint32_t width;
	uint32_t height;
	uint16_t planes;
	uint16_t bitsperpixel;
	uint32_t compression;
	uint32_t imagesize;
	uint32_t ypixelpermeter;
	uint32_t xpixelpermeter;
	uint32_t numcolorspallette;
	uint32_t mostimpcolor;
} bitmapinfoheader;

typedef struct {
	fileheader fileheader;
	bitmapinfoheader bitmapinfoheader;
} bitmap;

#pragma pack(pop)

int create_img(struct creature *creature);

#endif /* BMP_WRITER_H */