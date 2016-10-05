#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__
glm::vec3 calculateRandomDirectionInLobe(
glm::vec3 normal, float exponent, thrust::default_random_engine &rng) {
  thrust::uniform_real_distribution<float> u01(0, 1);

  float up = pow(sqrt(u01(rng)), 1.f / (exponent + 1)); // cos(theta)
  float over = sqrt(1 - up * up); // sin(theta)
  float around = u01(rng) * TWO_PI;

  // Find a direction that is not the normal based off of whether or not the
  // normal's components are all equal to sqrt(1/3) or whether or not at
  // least one component is less than sqrt(1/3). Learned this trick from
  // Peter Kutz.

  glm::vec3 directionNotNormal;
  if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
    directionNotNormal = glm::vec3(1, 0, 0);
  }
  else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
    directionNotNormal = glm::vec3(0, 1, 0);
  }
  else {
    directionNotNormal = glm::vec3(0, 0, 1);
  }

  // Use not-normal direction to generate two perpendicular directions
  glm::vec3 perpendicularDirection1 =
    glm::normalize(glm::cross(normal, directionNotNormal));
  glm::vec3 perpendicularDirection2 =
    glm::normalize(glm::cross(normal, perpendicularDirection1));

  return up * normal
    + cos(around) * over * perpendicularDirection1
    + sin(around) * over * perpendicularDirection2;
}

__host__ __device__
glm::vec3 calculateRandomDirectionInSteradianLobe(
glm::vec3 normal, float exponent, thrust::default_random_engine &rng, int idx, int N) {
  int idx2 = idx / N;
  idx = idx % N;
  thrust::uniform_real_distribution<float> u1(idx, idx + 1);
  thrust::uniform_real_distribution<float> u2(idx2, idx2 + 1);

  float up = pow(sqrt(u1(rng)), 1.f/(exponent + 1)); // cos(theta)
  float over = sqrt(1 - up * up); // sin(theta)
  float around = u2(rng) * TWO_PI;

  // Find a direction that is not the normal based off of whether or not the
  // normal's components are all equal to sqrt(1/3) or whether or not at
  // least one component is less than sqrt(1/3). Learned this trick from
  // Peter Kutz.

  glm::vec3 directionNotNormal;
  if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
    directionNotNormal = glm::vec3(1, 0, 0);
  }
  else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
    directionNotNormal = glm::vec3(0, 1, 0);
  }
  else {
    directionNotNormal = glm::vec3(0, 0, 1);
  }

  // Use not-normal direction to generate two perpendicular directions
  glm::vec3 perpendicularDirection1 =
    glm::normalize(glm::cross(normal, directionNotNormal));
  glm::vec3 perpendicularDirection2 =
    glm::normalize(glm::cross(normal, perpendicularDirection1));

  return up * normal
    + cos(around) * over * perpendicularDirection1
    + sin(around) * over * perpendicularDirection2;
}

__host__ __device__
glm::vec3 calculateRandomDirectionInSteradian(glm::vec3 normal, thrust::default_random_engine &rng, int idx, int N) {
  int idx2 = idx / N;
  idx = idx % N;
  thrust::uniform_real_distribution<float> u1(idx, idx + 1);
  thrust::uniform_real_distribution<float> u2(idx2, idx2 + 1);
  float up = sqrt(u1(rng) / (float)N); // cos(theta)
  float over = sqrt(1 - up * up); // sin(theta)
  float around = u2(rng) / (float)N * TWO_PI;

  glm::vec3 directionNotNormal;
  if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
    directionNotNormal = glm::vec3(1, 0, 0);
  }
  else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
    directionNotNormal = glm::vec3(0, 1, 0);
  }
  else {
    directionNotNormal = glm::vec3(0, 0, 1);
  }

  // Use not-normal direction to generate two perpendicular directions
  glm::vec3 perpendicularDirection1 =
    glm::normalize(glm::cross(normal, directionNotNormal));
  glm::vec3 perpendicularDirection2 =
    glm::normalize(glm::cross(normal, perpendicularDirection1));

  return up * normal
    + cos(around) * over * perpendicularDirection1
    + sin(around) * over * perpendicularDirection2;
}

namespace bxdf {
  namespace lambert {
    __host__ __device__ float pdf(const glm::vec3 &in, const glm::vec3 &normal, const glm::vec3 &out) {
      return 1 / 3.14159265;
    }
    __host__ __device__ glm::vec3 evaluateScatteredEnergy(const Material &m, const glm::vec3 &in, const glm::vec3 &normal, const glm::vec3 &out) {
      return m.color / 3.14159265f;
    }
    __host__ __device__ glm::vec3 sampleAndEvaluateScatteredEnergy(const Material &m, const glm::vec3 &in, const glm::vec3 &normal, glm::vec3 &out, float &pdf, thrust::default_random_engine &rng, int depth, int iter) {
      pdf = 1.f / 3.14159265f;

      if (depth == 0) {
        out = calculateRandomDirectionInSteradian(normal, rng, iter % (16 * 16), 16);
      }
      else {
        out = calculateRandomDirectionInHemisphere(normal, rng);
      }

      if (m.indexOfRefraction > 0) {
        glm::vec3 refr = glm::refract(in, normal, 1.f / m.indexOfRefraction);

        float fr = 0.5f * (glm::pow((m.indexOfRefraction * glm::abs(glm::dot(in, normal)) - glm::abs(glm::dot(refr, normal))) /
          (m.indexOfRefraction * glm::abs(glm::dot(in, normal)) + glm::abs(glm::dot(refr, normal))), 2) +
          glm::pow((glm::abs(glm::dot(in, normal)) - m.indexOfRefraction * glm::abs(glm::dot(refr, normal))) /
          (glm::abs(glm::dot(in, normal)) + m.indexOfRefraction * glm::abs(glm::dot(refr, normal))), 2));

        thrust::uniform_real_distribution<float> u;
        float fac = u(rng);

        if (fac < fr) {
          out = glm::reflect(in, normal);
          return m.specular.color / 3.14159265f;
        } else {
          return m.color / 3.14159265f;
        }
      } else {
        return m.color / 3.14159265f;
      }
    }
  }

  namespace mirror {
    __host__ __device__ float pdf(const glm::vec3 &in, const glm::vec3 &normal, const glm::vec3 &out) {
      return 0;
    }
    __host__ __device__ glm::vec3 evaluateScatteredEnergy(const Material &m, const glm::vec3 &in, const glm::vec3 &normal, const glm::vec3 &out) {
      return m.specular.color;
    }
    __host__ __device__ glm::vec3 sampleAndEvaluateScatteredEnergy(const Material &m, const glm::vec3 &in, const glm::vec3 &normal, glm::vec3 &out, float &pdf, thrust::default_random_engine &rng, int depth, int iter) {
      out = glm::reflect(in, normal);
      pdf = 1;
      return m.specular.color;
    }
  }

  namespace glass {
    __host__ __device__ float pdf(const glm::vec3 &in, const glm::vec3 &normal, const glm::vec3 &out) {
      return 0;
    }
    __host__ __device__ glm::vec3 evaluateScatteredEnergy(const Material &m, const glm::vec3 &in, const glm::vec3 &normal, const glm::vec3 &out) {
      return m.specular.color;
    }
    __host__ __device__ glm::vec3 sampleAndEvaluateScatteredEnergy(const Material &m, const glm::vec3 &in, const glm::vec3 &normal, glm::vec3 &out, float &pdf, thrust::default_random_engine &rng, int depth, int iter) {
      pdf = 1;
      glm::vec3 norm = normal;
      float ior = m.indexOfRefraction;
      if (glm::dot(in, norm) > 0) {
        norm = -norm;
        ior = 1.f / ior;
      }

      glm::vec3 refl = glm::reflect(in, norm);
      glm::vec3 refr = glm::refract(in, norm, 1.f / ior);

      if (ior < 1 && glm::dot(in, -norm) < glm::cos(glm::asin(ior))) {
        out = refl;
      } else {
        float fr = 0.5f * (glm::pow((ior * glm::abs(glm::dot(in, norm)) - glm::abs(glm::dot(refr, norm))) /
          (ior * glm::abs(glm::dot(in, norm)) + glm::abs(glm::dot(refr, norm))), 2) +
          glm::pow((glm::abs(glm::dot(in, norm)) - ior * glm::abs(glm::dot(refr, norm))) /
          (glm::abs(glm::dot(in, norm)) + ior * glm::abs(glm::dot(refr, norm))), 2));

        thrust::uniform_real_distribution<float> u;
        float fac = u(rng);

        if (fac < fr) {
          out = refl;
        }
        else {
          out = refr;
        }
      }

      
      return m.specular.color;
    }
  }
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng, 
        int depth, int iter) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
  glm::vec3 out;
  float pdf;
  glm::vec3 col;
  const glm::vec3& in = pathSegment.ray.direction;
 
  /*if (m.hasReflective) {
    col = bxdf::mirror::sampleAndEvaluateScatteredEnergy(m, in, normal, out, pdf, rng, depth, iter);
  } else if (m.hasRefractive) {
    col = bxdf::glass::sampleAndEvaluateScatteredEnergy(m, in, normal, out, pdf, rng, depth, iter);
  } else {
    col = bxdf::lambert::sampleAndEvaluateScatteredEnergy(m, in, normal, out, pdf, rng, depth, iter);
  }*/


  glm::vec3 norm = normal;
  if (depth == 0) {
    norm = calculateRandomDirectionInSteradianLobe(normal, m.specular.exponent, rng, iter % (16 * 16), 16);
  }
  else {
    norm = calculateRandomDirectionInLobe(normal, m.specular.exponent, rng);
  }

  float ior = m.indexOfRefraction;
  if (glm::dot(in, norm) > 0) {
    norm = -norm;
    ior = 1.f / ior;
  }

  glm::vec3 refl = glm::reflect(in, norm);
  glm::vec3 refr = glm::refract(in, norm, 1.f / ior);

  float fr = 0.5f * (glm::pow((ior * glm::abs(glm::dot(in, norm)) - glm::abs(glm::dot(refr, norm))) /
    (ior * glm::abs(glm::dot(in, norm)) + glm::abs(glm::dot(refr, norm))), 2) +
    glm::pow((glm::abs(glm::dot(in, norm)) - ior * glm::abs(glm::dot(refr, norm))) /
    (glm::abs(glm::dot(in, norm)) + ior * glm::abs(glm::dot(refr, norm))), 2));

  float refl_fac = glm::min(1.f, fr + m.hasReflective);
  float refr_fac = (1.f - refl_fac) * m.hasRefractive;
  float diff_fac = (1.f - refl_fac) * (1.f - m.hasRefractive);

  thrust::uniform_real_distribution<float> u;
  float fac = u(rng);
  if (fac < refl_fac) {
    out = refl;
    glm::vec3 h = glm::normalize(out + -in);

    float G = min(1.f, min(
      2.f*glm::abs(glm::dot(normal, h))*glm::abs(glm::dot(normal, in)) / glm::abs(glm::dot(in, h)),
      2.f*glm::abs(glm::dot(normal, h))*glm::abs(glm::dot(normal, out)) / glm::abs(glm::dot(in, h))
      ));
    pdf = (m.specular.exponent + 1) * glm::pow(glm::abs(glm::dot(h, normal)), m.specular.exponent) / TWO_PI;
    
    col = m.specular.color * pdf * G;
  } else if (fac < refl_fac + refr_fac) {
    out = refr;
    pdf = 1.f;
    col = m.color;
  } else {
    if (depth == 0) {
      out = calculateRandomDirectionInSteradian(normal, rng, iter % (16 * 16), 16);
    }
    else {
      out = calculateRandomDirectionInHemisphere(normal, rng);
    }
    col = m.color / 3.14159265f;
    pdf = 1.f / 3.14159265f;
  }

  pathSegment.ray.direction = out;
  pathSegment.ray.origin = intersect + out * 0.01f;
  pathSegment.color *= col / pdf;
}
