module;
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
export module GFX.Transform;
import ECS.UID;

namespace SIByL::GFX
{
	export struct Transform
	{
	public:
		auto getTranslation() const noexcept -> glm::vec3 { return translation; }
		auto getEulerAngles() const noexcept -> glm::vec3 { return eulerAngles; }
		auto getScale() const noexcept -> glm::vec3 { return scale; }

		auto setTranslation(glm::vec3 const& input) noexcept -> void { translation = input; }
		auto setEulerAngles(glm::vec3 const& input) noexcept -> void { eulerAngles = input; }
		auto setScale(glm::vec3 const& input) noexcept -> void { scale = input; }

		auto getTransform() noexcept -> glm::mat4x4 { return transform; }
		auto getRotatedForward() noexcept -> glm::vec3;
		auto invalidTransform() noexcept -> glm::mat4x4;

		auto getAccumulativeTransform() noexcept -> glm::mat4x4 { return accumulativeTransform; }
		auto propagateFromPrecursor(glm::mat4x4 const& precursor_transform) noexcept;

	private:
		glm::vec3 translation = { 0.0f, 0.0f, 0.0f };
		glm::vec3 eulerAngles = { 0.0f, 0.0f, 0.0f };
		glm::vec3 scale = { 1.0f, 1.0f, 1.0f };
		glm::mat4x4 transform = invalidTransform();
		glm::mat4x4 precursorTransform;
		glm::mat4x4 accumulativeTransform;
	};

	auto Transform::invalidTransform() noexcept -> glm::mat4x4
	{
		glm::mat4 rotation = glm::toMat4(glm::quat(eulerAngles));
		return glm::translate(glm::mat4(1.0f), translation)
			* rotation
			* glm::scale(glm::mat4(1.0f), scale);
	}

	auto Transform::getRotatedForward() noexcept -> glm::vec3
	{
		glm::mat3 rotation = glm::toMat3(glm::quat(eulerAngles));
		return rotation * glm::vec3(0, 0, 1);
	}

	auto Transform::propagateFromPrecursor(glm::mat4x4 const& precursor_transform) noexcept
	{
		precursorTransform = precursor_transform;
		accumulativeTransform = getTransform() * accumulativeTransform;
	}
}