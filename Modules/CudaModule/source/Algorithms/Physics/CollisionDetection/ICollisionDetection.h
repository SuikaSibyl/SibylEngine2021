#pragma once

namespace SIByL
{
	class ICollisionDetection
	{
	public:
		static void SelfCollisionDetect(const std::vector<float>& vertices, const std::vector<unsigned int>& indices,
			std::vector<std::pair<int, int>>& collided_pairs);
	};
}