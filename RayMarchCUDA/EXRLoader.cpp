#include <OpenEXR/ImfArray.h>
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfInputFile.h>

void
readRgba(const char fileName[],
	int& width,
	int& height,
	float* &data)
{
	Imf::Array2D<Imf::Rgba> pixels;
	Imf::RgbaInputFile file(fileName);
	Imath::Box2i dw = file.dataWindow();
	width = dw.max.x - dw.min.x + 1;
	height = dw.max.y - dw.min.y + 1;
	pixels.resizeErase(height, width);
	file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * width, 1, width);
	file.readPixels(dw.min.y, dw.max.y);
	data = new float[width * height * sizeof(float) * 4] ;
	memcpy(data,&pixels[0][0], width * height * sizeof(float)*2);
}

inline float inverse(float x) { return 1.0f / x; }

void
readGZ(const char fileName[],
	int& width, int& height, float*& data, int Zmode)
{
	Imf::Array2D<float> rPixels;
	Imf::Array2D<float> gPixels;
	Imf::Array2D<float> bPixels;
	//Imf::Array2D<float> aPixels;
	//Imf::Array2D<unsigned int> zPixels;
	Imf::Array2D<float> zPixels;

	Imf::InputFile file(fileName);
	Imath::Box2i dw = file.header().dataWindow();
	width = dw.max.x - dw.min.x + 1;
	height = dw.max.y - dw.min.y + 1;
	rPixels.resizeErase(height, width);
	gPixels.resizeErase(height, width);
	bPixels.resizeErase(height, width);
	//aPixels.resizeErase(height, width);
	zPixels.resizeErase(height, width);
	Imf::FrameBuffer frameBuffer;
	frameBuffer.insert("R", // name
		Imf::Slice(Imf::FLOAT, // type
			(char*)(&rPixels[0][0] - dw.min.x - dw.min.y * width),// base
			sizeof(float) * 1, // xStride
			sizeof(float) * width,// yStride
			1, 1, // x/y sampling
			0.0f)); // fillValue
	frameBuffer.insert("G", // name
		Imf::Slice(Imf::FLOAT, // type
			(char*)(&gPixels[0][0] - dw.min.x - dw.min.y * width),// base
			sizeof(float) * 1, // xStride
			sizeof(float) * width,// yStride
			1, 1, // x/y sampling
			0.0f)); // fillValue
	frameBuffer.insert("B", // name
		Imf::Slice(Imf::FLOAT, // type
			(char*)(&bPixels[0][0] - dw.min.x - dw.min.y * width),// base
			sizeof(float) * 1, // xStride
			sizeof(float) * width,// yStride
			1, 1, // x/y sampling
			0.0f)); // fillValue
	/*frameBuffer.insert("A", // name
		Imf::Slice(Imf::FLOAT, // type
			(char*)(&aPixels[0][0] - dw.min.x - dw.min.y * width),// base
			sizeof(float) * 1, // xStride
			sizeof(float) * width,// yStride
			1, 1, // x/y sampling
			0.0f)); // fillValue*/
	/*frameBuffer.insert("Z", // name
		Imf::Slice(Imf::UINT, // type
			(char*)(&zPixels[0][0] - dw.min.x - dw.min.y * width),// base	
			sizeof(zPixels[0][0]) * 1, // xStride
			sizeof(zPixels[0][0]) * width,// yStride
			1, 1, // x/y sampling
			UINT_MAX)); // fillValue*/
	
	/*
	switch (Zmode) {
	case 1: {
		frameBuffer.insert("Z", // name
		Imf::Slice(Imf::FLOAT, // type
			(char*)(&zPixels[0][0] - dw.min.x - dw.min.y * width),// base	
			sizeof(zPixels[0][0]) * 1, // xStride
			sizeof(zPixels[0][0]) * width,// yStride
			1, 1, // x/y sampling
			0.0f)); // fillValue
		break; 
	};
	case 0: {	*/
		frameBuffer.insert("A", // name
		Imf::Slice(Imf::FLOAT, // type
			(char*)(&zPixels[0][0] - dw.min.x - dw.min.y * width),// base
			sizeof(zPixels[0][0]) * 1, // xStride
			sizeof(zPixels[0][0]) * width,// yStride
			1, 1, // x/y sampling
			0.0f)); // fillValue
	//	break; 
	//};
	//}
	/**frameBuffer.insert("Z", // name
		Imf::Slice(Imf::FLOAT, // type
			(char*)(&zPixels[0][0] - dw.min.x - dw.min.y * width),// base	
			sizeof(zPixels[0][0]) * 1, // xStride
			sizeof(zPixels[0][0]) * width,// yStride
			1, 1, // x/y sampling
			0.0f)); // fillValue*/
	file.setFrameBuffer(frameBuffer);
	file.readPixels(dw.min.y, dw.max.y);
	data = new float[width * height * sizeof(float) * 4];
	float* ptr = data;
	for (int i = 0; i < width * height; ++i) {
		*ptr++ = *(&rPixels[0][0] + i);
		*ptr++ = *(&gPixels[0][0] + i);
		*ptr++ = *(&bPixels[0][0] + i);
		//*ptr++ = -*(&zPixels[0][0] + i);
		switch (Zmode) { 
		case 1: {*ptr++ = inverse(*(&zPixels[0][0] + i)); break; };
		case 0: {*ptr++ = *(&zPixels[0][0] + i); break; };
		}
		
		//*ptr++ = (float)(((double)(*(&zPixels[0][0] + i)) / double(UINT32_MAX))*255.0f);
		//*ptr++ = 0.5f;
	}

	/*unsigned int Mx= 0;
	unsigned int Mn = UINT32_MAX;
	for (int i = 0; i < width * height; ++i) {
		unsigned int tmp = *(&zPixels[0][0]+i);
		if (tmp > Mx) Mx = tmp;
		if (tmp < Mn) Mn = tmp;
	}
	std::cout << Mn <<" "<< Mx << std::endl;*/
}
