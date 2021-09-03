#pragma once

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace SIByL
{
	struct TransformComponent
	{
	public:
		glm::vec3 Translation = { 0.0f, 0.0f, 0.0f };
		glm::vec3 EulerAngles = { 0.0f, 0.0f, 0.0f };
		glm::vec3 Scale = { 0.0f, 0.0f, 0.0f };

		//TransformComponent() = default;
		TransformComponent(const TransformComponent&) = default;
		TransformComponent(
			const glm::vec3& translation = {0,0,0}, 
			const glm::vec3& rotate = { 0,0,0 }, 
			const glm::vec3& scale = { 0,0,0 })
			: Translation(translation), EulerAngles(rotate), Scale(scale) {}

		glm::mat4 GetTransform() const
		{
			glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), EulerAngles.x, { 1,0,0 })
				* glm::rotate(glm::mat4(1.0f), EulerAngles.y, { 0,1,0 })
				* glm::rotate(glm::mat4(1.0f), EulerAngles.z, { 0,0,1 });

			return glm::translate(glm::mat4(1.0f), Translation)
				* rotation
				* glm::scale(glm::mat4(1.0f), Scale);
		}
	};
}