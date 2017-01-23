#include "bmp_writer.h"

int create_img(struct creature *creature) {
	int i, j;
	FILE *fp = fopen("test.bmp", "wb");
	bitmap *pbitmap = (bitmap*)calloc(1, sizeof(bitmap));
	uint8_t *pixelbuffer = (uint8_t*)malloc(_pixelbytesize);
	pbitmap->fileheader.signature[0] = 0x42;
	pbitmap->fileheader.signature[1] = 0x4D;
	pbitmap->fileheader.filesize = _filesize;
	pbitmap->fileheader.fileoffset_to_pixelarray = sizeof(bitmap);
	pbitmap->bitmapinfoheader.dibheadersize = sizeof(bitmapinfoheader);
	pbitmap->bitmapinfoheader.width = _width;
	pbitmap->bitmapinfoheader.height = _height;
	pbitmap->bitmapinfoheader.planes = _planes;
	pbitmap->bitmapinfoheader.bitsperpixel = _bitsperpixel;
	pbitmap->bitmapinfoheader.compression = _compression;
	pbitmap->bitmapinfoheader.imagesize = _pixelbytesize;
	pbitmap->bitmapinfoheader.ypixelpermeter = _ypixelpermeter;
	pbitmap->bitmapinfoheader.xpixelpermeter = _xpixelpermeter;
	pbitmap->bitmapinfoheader.numcolorspallette = 0;
	fwrite(pbitmap, 1, sizeof(bitmap), fp);
	memset(pixelbuffer, 1, _pixelbytesize);
	for(i = 0; i < creature->n; i++){
		for(j = 0; j < creature->n; j++){
			printf("%d %d %d\n", creature->cells[i * creature->n + j].v[2], creature->cells[i * creature->n + j].v[3], creature->cells[i * creature->n + j].v[4]);
			pixelbuffer[i * _width + j] = (creature->cells[i * creature->n + j].v[2] + creature->cells[i * creature->n + j].v[3] + creature->cells[i * creature->n + j].v[4])/3;
		}
	}
	fwrite(pixelbuffer, 1, _pixelbytesize, fp);
	fclose(fp);
	free(pbitmap);
	free(pixelbuffer);
	return 0;
}