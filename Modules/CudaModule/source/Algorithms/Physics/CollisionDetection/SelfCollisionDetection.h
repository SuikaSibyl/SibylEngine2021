#pragma once

namespace SIByL
{
	namespace CUDA
	{
		class SelfCollisionDetection
		{
		public:
			void FindCollision(const std::vector<float>& vertices, const std::vector<unsigned int>& indices,
				std::vector<std::pair<int, int>>& collided_pairs);

		private:

		};
	}
}