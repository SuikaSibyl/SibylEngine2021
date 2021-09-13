#include "SIByLpch.h"
#include "MathUtils.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

namespace SIByL
{
	bool DecomposeTransform(const glm::mat4& transform, glm::vec3& outTranslation, glm::vec3& outRotation, glm::vec3& outScale)
	{
		using namespace glm;
		using T = float;

		mat4 LocalMatrix(transform);

		return true;
	}
}