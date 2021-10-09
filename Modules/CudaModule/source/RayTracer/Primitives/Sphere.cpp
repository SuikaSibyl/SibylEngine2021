#include "helper_math.h"
#include "CudaModulePCH.h"

#include "Sphere.h"
#include "RayTracer/Core/Ray.h"

__device__ CUDASphere::CUDASphere(float3 c, float r, CUDAMaterial* mat)
    :CUDAHitable(mat), center(c), radius(r)
{

}

__device__ bool CUDASphere::hit(const CUDARay& r, float t_min, float t_max, CUDAHitRecord& rec) const
{
    float3 connection = center - r.origin;
    float distance = length(connection);

    // Case 1
    // The start-point of the ray is inside the sphere
    // Absolutely 1 intersection would happen
    bool inside = distance < radius;

    // Case 2
    // The start-point of the ray is outside the sphere
    float projection = dot(connection, r.direction);
    // If the sphere is behind the ray, no intersection would happen
    if ((projection < t_min) && (!inside))
        return false;

    // Case 3
    float3 projected = r.pointAtParameter(projection);
    // If the sphere is too far away from the ray
    float mindis = length(center - projected);
    if ((mindis > radius) && (!inside))
        return false;

	// Case 4
	// Absolutely 2 intersection would happen
	float nearest = (float)sqrt(radius * radius - mindis * mindis);
	float minus = projection - nearest;
	if (minus > t_min)
		nearest = minus;
	else
		nearest = projection + nearest;

	if (nearest < rec.t)
	{
        float3 hitpoint = r.pointAtParameter(nearest);
        rec.normal = normalize(hitpoint - center);
        rec.t = nearest;
        rec.p = hitpoint;
	}

	// Do intersect
	return true;
}