#include "CudaModulePCH.h"
#include "ICollisionDetection.h"

#include "SelfCollisionDetection.h"

namespace SIByL
{
	void ICollisionDetection::SelfCollisionDetect(const std::vector<float>& vertices, const std::vector<unsigned int>& indices,
		std::vector<std::pair<int, int>>& collided_pairs)
	{
		CUDA::SelfCollisionDetection detector;
		detector.FindCollision(vertices, indices, collided_pairs);
	}

}