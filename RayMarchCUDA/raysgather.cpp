#include <vector_types.h>
#include "raysgather.h"

int gatherRays(float4* IR, float4* IR_out, int N) {
	int j = 0;
	for (int i = 0; i < N; ++i) {
		float4 tmp = IR[i * 2 + 1];
		if (tmp.w > 0.0f) {
			IR_out[j * 2] = IR[i * 2];
			IR_out[j * 2 + 1] = IR[i * 2 + 1];
			++j;
		}
	}
	return j;
}