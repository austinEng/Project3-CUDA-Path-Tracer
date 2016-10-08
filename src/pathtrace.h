#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

typedef struct Uniform {
  float pixel_x;
  float pixel_y;
  float hemi_1;
  float hemi_2;
  float mat;
  float light;
  float light1;
  float light2;
} Uniform;