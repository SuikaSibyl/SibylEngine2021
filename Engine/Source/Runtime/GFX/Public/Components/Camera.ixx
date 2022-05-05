module;
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
export module GFX.Camera;
import ECS.UID;

namespace SIByL::GFX
{
	export enum struct ProjectKind
	{
		PERSPECTIVE,
		ORTHOGONAL,
	};

	export struct Camera
	{
	public:
		auto invalidViewMat() noexcept -> glm::mat4x4;
		auto invalidProjectionMat() noexcept -> glm::mat4x4;
		
		auto getFovy() noexcept -> float { return fovy; }
		auto getAspect() noexcept -> float { return aspect; }
		auto getNear() noexcept -> float { return near; }
		auto getFar() noexcept -> float { return far; }

		auto setFovy(float input) noexcept -> void { fovy = input; }
		auto setAspect(float input) noexcept -> void { aspect = input; }
		auto setNear(float input) noexcept -> void {  near = input; }
		auto setFar(float input) noexcept -> void { far = input; }

		ProjectKind kind = ProjectKind::PERSPECTIVE;

	private:
		float fovy = glm::radians(45.0f);
		float aspect = 1;
		float near = 0.1f;
		float far = 100.0f;

		float left_right = 0;
		float bottom_top = 0;

	private:
		glm::mat4x4 view;
		glm::mat4x4 projection;
	};

	auto Camera::invalidViewMat() noexcept -> glm::mat4x4
	{
		return view;
	}

	auto Camera::invalidProjectionMat() noexcept -> glm::mat4x4
	{
		if (kind == ProjectKind::PERSPECTIVE)
		{
			projection = glm::perspective(fovy, aspect, near, far);
		}
		else if (kind == ProjectKind::ORTHOGONAL)
		{
			projection = glm::ortho(-left_right, left_right, -bottom_top, bottom_top, near, far);
		}
		return projection;
	}
}