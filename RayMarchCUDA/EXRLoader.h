#pragma once

extern "C++" void
readRgba(const char fileName[],
	int& width,
	int& height,
	float*& data);

/*extern "C++"*/ void
readGZ(const char fileName[],
	int& width, int& height, float*& data, int Zmode = 1);