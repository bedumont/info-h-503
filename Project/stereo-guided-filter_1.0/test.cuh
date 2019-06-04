#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "costVolume.h"
#include "image.h"
#include <iostream>

void cuTest();
Image fcv(float* im1Color, float* im2Color, int w, int h,
	int dispMin, int dispMax, const ParamGuidedFilter& param);