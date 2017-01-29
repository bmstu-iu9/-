#include "bmp_writer.h"

int create_img(struct creature *creature) {
	FILE *fp = fopen("test.bmp", "wb");
	bitmap *pbitmap = (bitmap*)calloc(1, sizeof(bitmap));
	uint8_t *pixelbuffer = (uint8_t*)malloc(_pixelbytesize);
	pbitmap->fileheader.signature[0] = 0x42;
	pbitmap->fileheader.signature[1] = 0x4D;
	pbitmap->fileheader.filesize = _filesize;
	pbitmap->fileheader.fileoffset_to_pixelarray = sizeof(bitmap);
	pbitmap->bitmapinfoheader.dibheadersize = sizeof(bitmapinfoheader);
	pbitmap->bitmapinfoheader.width = creature->n;
	pbitmap->bitmapinfoheader.height = creature->n;
	pbitmap->bitmapinfoheader.planes = _planes;
	pbitmap->bitmapinfoheader.bitsperpixel = _bitsperpixel;
	pbitmap->bitmapinfoheader.compression = _compression;
	pbitmap->bitmapinfoheader.imagesize = _pixelbytesize;
	pbitmap->bitmapinfoheader.ypixelpermeter = _ypixelpermeter;
	pbitmap->bitmapinfoheader.xpixelpermeter = _xpixelpermeter;
	pbitmap->bitmapinfoheader.numcolorspallette = 0;
	fwrite(pbitmap, 1, sizeof(bitmap), fp);
	//memset(pixelbuffer, 1, _pixelbytesize);
	for(int i = 0; i < creature->n; i++){
		for(int j = 0; j < creature->n; j++){
			//printf("%d %d %d\n", creature->cells[i * creature->n + j].v[2], creature->cells[i * creature->n + j].v[3], creature->cells[i * creature->n + j].v[4]);
			pixelbuffer[(i * creature->n + j) * _bitsperpixel/8] = creature->cells[i * creature->n + j].v[2];
			pixelbuffer[(i * creature->n + j) * _bitsperpixel/8 + 1] = creature->cells[i * creature->n + j].v[3];
			pixelbuffer[(i * creature->n + j) * _bitsperpixel/8 + 2] = creature->cells[i * creature->n + j].v[4];
		}
	}
	fwrite(pixelbuffer, 1, _pixelbytesize, fp);
	fclose(fp);	
	free(pbitmap);
	free(pixelbuffer);
	return 0;
}