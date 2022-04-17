module;
#include <glm/glm.hpp>
export module Math.LinearAlgebra;

namespace SIByL::Math
{
	inline auto decomposeTransform(const glm::mat4& transform, glm::vec3& outTranslation, glm::vec3& outRotation, glm::vec3& outScale) noexcept -> bool;
}