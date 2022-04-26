module;
#include <glm/glm.hpp>
#include <vector>
export module GFX.BoundingBox;

namespace SIByL::GFX
{
	export struct BoundingBox
	{
		glm::vec4 min = { 0,0,0,0 };
		glm::vec4 max = { 0,0,0,0 };

		auto getNode(int i) noexcept -> glm::vec3;
	};

	export auto getBBoxLinePairs() noexcept -> std::vector<int>
	{
		static std::vector<int> pairs =
		{
			// bottom
			0,1,
			1,3,
			3,2,
			0,2,
			// up
			4,5,
			5,7,
			7,6,
			4,6,
			// mid
			0,4,
			1,5,
			3,7,
			2,6,
		};
		return pairs;
	}

	auto BoundingBox::getNode(int i) noexcept -> glm::vec3
	{
		int x_val = i % 2;
		int y_val = int(i / 2) % 2;
		int z_val = int(i / 4) % 2;

		float x = x_val == 1 ? max.x : min.x;
		float y = y_val == 1 ? max.y : min.y;
		float z = z_val == 1 ? max.z : min.z;
		return glm::vec3(x, y, z);
	}

}