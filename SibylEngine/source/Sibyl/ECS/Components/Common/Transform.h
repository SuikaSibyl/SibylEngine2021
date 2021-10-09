#pragma once

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>

namespace SIByL
{
	struct TransformComponent
	{
	public:
		friend class Scene;

	public:
		glm::vec3 Translation = { 0.0f, 0.0f, 0.0f };
		glm::vec3 EulerAngles = { 0.0f, 0.0f, 0.0f };
		glm::vec3 Scale = { 1.0f, 1.0f, 1.0f };

		TransformComponent(const TransformComponent&) = default;
		TransformComponent(
			const glm::vec3& translation = {0,0,0}, 
			const glm::vec3& rotate = { 0,0,0 }, 
			const glm::vec3& scale = { 1,1,1 })
			: Translation(translation), EulerAngles(rotate), Scale(scale) {}

		glm::mat4 GetTransform() const
		{
			glm::mat4 rotation = glm::toMat4(glm::quat(EulerAngles));

			return glm::translate(glm::mat4(1.0f), Translation)
				* rotation
				* glm::scale(glm::mat4(1.0f), Scale);
		}

		void SetParent(const uint64_t& p);
		const uint64_t& GetParent() { return parent; }
		uint64_t GetUid() { return uid; }
		void SetUid(const uint64_t& u) { uid = u; }
		std::vector<uint64_t>& GetChildren() { return children; }

	private:
		void AddChild(const uint64_t& c);
		void RemoveChild(const uint64_t& c);

	private:
		uint64_t parent = 0;
		uint64_t uid = 0;
		Scene* scene = nullptr;
		std::vector<uint64_t> children;
	};
}