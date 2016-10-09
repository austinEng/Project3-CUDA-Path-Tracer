#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <curand.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
// #define PROFILING

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static float * dev_pdfTotal = NULL;
static Uniform * dev_uniforms = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static int * dev_lightIds = NULL;
static int nLights;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static FirstIntersection * dev_first_intersections = NULL;
static curandGenerator_t qrng;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_uniforms, pixelcount * sizeof(Uniform));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
    
  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_first_intersections, pixelcount * sizeof(FirstIntersection));
    cudaMemset(dev_first_intersections, 0, pixelcount * sizeof(FirstIntersection));

    // TODO: initialize any extra device memeory you need
      
    curandCreateGenerator(&qrng, CURAND_RNG_QUASI_SOBOL32);
    curandSetQuasiRandomGeneratorDimensions(qrng, sizeof(Uniform) / sizeof(float));
    curandSetGeneratorOrdering(qrng, CURAND_ORDERING_QUASI_DEFAULT);

    std::vector<int> lightIds;
    for (size_t i = 0; i < scene->geoms.size(); ++i) {
      Geom &geo = scene->geoms.at(i);
      int mid = geo.materialid;
      Material &mat = scene->materials.at(mid);
      if (mat.emittance > 0) {
        lightIds.push_back(i);
      }
    }
    nLights = lightIds.size();
    cudaMalloc(&dev_lightIds, lightIds.size() * sizeof(int));
    cudaMemcpy(dev_lightIds, lightIds.data(), lightIds.size() * sizeof(int), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_uniforms);
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    cudaFree(dev_first_intersections);
    cudaFree(dev_lightIds);
    // TODO: clean up any extra device memory you created

    curandDestroyGenerator(qrng);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, float* uX, float* uY)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
    segment.ray.origin += uX[index] * cam.pixelLength.x * cam.right;
    segment.ray.origin += uY[index] * cam.pixelLength.y * cam.up;
    segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
    segment.pdf = 1;

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
  int iter
	, int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
  , FirstIntersection * first_intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
    if (depth == 0) {
      if (iter == 0 ||
        pathSegments[path_index].ray.origin.x != first_intersections[path_index].x ||
        pathSegments[path_index].ray.origin.y != first_intersections[path_index].y
      ) {
        intersectPath(pathSegments[path_index], geoms, geoms_size, first_intersections[path_index].intersection);
        first_intersections[path_index].x = pathSegments[path_index].ray.origin.x;
        first_intersections[path_index].y = pathSegments[path_index].ray.origin.y;
      }
      intersections[path_index] = first_intersections[path_index].intersection;
    } else {
      intersectPath(pathSegments[path_index], geoms, geoms_size, intersections[path_index]);
    }
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.

__global__ void shadeMaterial(
  int iter
  , int num_paths
  , ShadeableIntersection * shadeableIntersections
  , PathSegment * pathSegments
  , Material * materials
  , Geom * geoms
  , int * lightIds
  , int nLights
  , int geoms_size
  , int depth
  , float * u_hemi1
  , float * u_hemi2
  , float * u_mat
  , float * u_light
  , float * u_light1
  , float * u_light2
  )
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    if (!pathSegments[idx].remainingBounces) return;
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      
      //thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      //thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
        pathSegments[idx].remainingBounces = 0;
      }
      else {
        pathSegments[idx].remainingBounces--;
        if (pathSegments[idx].remainingBounces == 0) {
          pathSegments[idx].color = glm::vec3(0.0f);
          return;
        }
        glm::vec3 brdfcol;
        float brdfpdf = 0;
        glm::vec3 intersect = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t;
        scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, u_hemi1[idx], u_hemi2[idx], u_mat[idx], brdfcol, brdfpdf);

        glm::vec3 lightcol;
        float lightpdf = 0;
        glm::vec3 pt;
        const Geom &light = geoms[lightIds[(int)(u_light[idx] * nLights)]];
        if (light.type == SPHERE) {
          // samplePointOnSphere(light, u_light1[idx], u_light2[idx], lightpdf, intersect, pt);
        }
        else if (light.type == CUBE) {
          // samplePointOnCube(light, u_light1[idx], u_light2[idx], lightpdf, intersect, pt);
        }

        pathSegments[idx].color *=
          brdfpdf / (brdfpdf + lightpdf) * brdfcol +
          lightpdf / (brdfpdf + lightpdf) * lightcol;
      }
    }
    else {
      pathSegments[idx].color = glm::vec3(0.0f);
      pathSegments[idx].remainingBounces = 0;
    }
  }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, float * pdfs, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
    image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

struct filterPaths
{
    __host__ __device__
    bool operator()(const PathSegment& segment) {
      return segment.remainingBounces > 0;
    }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	  // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

  float* u_pixelX;
  float* u_pixelY;
  float* u_hemi1;
  float* u_hemi2;
  float* u_mat;
  float* u_light;
  float* u_light1;
  float* u_light2;

  curandGenerateUniform(qrng, (float*)dev_uniforms, sizeof(Uniform) / sizeof(float) * pixelcount);
  u_pixelX = (float*)dev_uniforms + offsetof(Uniform, pixel_x) / sizeof(float) * pixelcount;
  u_pixelY = (float*)dev_uniforms + offsetof(Uniform, pixel_y) / sizeof(float) * pixelcount;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths, u_pixelX, u_pixelY);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
	while (!iterationComplete) {
    cudaEvent_t start, stop;
    float milliseconds = 0;
#ifdef PROFILING
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    curandGenerateUniform(qrng, (float*)dev_uniforms, sizeof(Uniform) / sizeof(float) * pixelcount);
#ifdef PROFILING
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    ofstream curandTime;
    curandTime.open("curand.txt", std::ios::app);
    curandTime << milliseconds << "ms\n";
    curandTime.close();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

    u_pixelX = (float*)dev_uniforms + offsetof(Uniform, pixel_x) / sizeof(float) * pixelcount;
    u_pixelY = (float*)dev_uniforms + offsetof(Uniform, pixel_y) / sizeof(float) * pixelcount;
    u_hemi1 = (float*)dev_uniforms + offsetof(Uniform, hemi_1) / sizeof(float) * pixelcount;
    u_hemi2 = (float*)dev_uniforms + offsetof(Uniform, hemi_2) / sizeof(float) * pixelcount;
    u_mat = (float*)dev_uniforms + offsetof(Uniform, mat) / sizeof(float) * pixelcount;
    u_light = (float*)dev_uniforms + offsetof(Uniform, light) / sizeof(float)* pixelcount;
    u_light1 = (float*)dev_uniforms + offsetof(Uniform, light1) / sizeof(float)* pixelcount;
    u_light2 = (float*)dev_uniforms + offsetof(Uniform, light2) / sizeof(float)* pixelcount;

	  // clean shading chunks
	  cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	  // tracing
	  dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#ifdef PROFILING
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
      iter
		  , depth
		  , num_paths
		  , dev_paths
		  , dev_geoms
		  , hst_scene->geoms.size()
		  , dev_intersections
      , dev_first_intersections
		  );
#ifdef PROFILING
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    ofstream intersectTime;
    intersectTime.open("intersections.txt", std::ios::app);
    intersectTime << milliseconds << "ms\n";
    intersectTime.close();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

	  // TODO:
	  // --- Shading Stage ---
	  // Shade path segments based on intersections and generate new rays by
    // evaluating the BSDF.
    // Start off with just a big kernel that handles all the different
    // materials you have in the scenefile.
    // TODO: compare between directly shading the path segments and shading
    // path segments that have been reshuffled to be contiguous in memory.

#ifdef PROFILING
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
      iter,
      num_paths,
      dev_intersections,
      dev_paths,
      dev_materials,
      dev_geoms,
      dev_lightIds,
      nLights,
		  hst_scene->geoms.size(),
      depth,
      u_hemi1, u_hemi2, u_mat,
      u_light, u_light1, u_light2
    );
#ifdef PROFILING
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    ofstream shadeTime;
    shadeTime.open("shadeThrust.txt", std::ios::app);
    shadeTime << milliseconds << "ms\n";
    shadeTime.close();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
    
#ifdef PROFILING
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    PathSegment* end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, filterPaths());
#ifdef PROFILING
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    ofstream compactTime;
    compactTime.open("compact.txt", std::ios::app);
    compactTime << milliseconds << "ms\n";
    compactTime.close();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
    num_paths = std::distance(dev_paths, end);
    if (num_paths == 0) {
      iterationComplete = true;
      num_paths = std::distance(dev_paths, dev_path_end);
      depth = 0;
    } 

    checkCUDAError("trace one bounce");
    cudaDeviceSynchronize();
    depth++;
	}

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_pdfTotal, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
