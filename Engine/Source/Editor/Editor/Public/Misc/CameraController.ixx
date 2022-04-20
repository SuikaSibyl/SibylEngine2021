module;
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <imgui.h>
#include <imgui_internal.h>
#include <functional>
#include "entt.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>
export module Editor.CameraController;
import GFX.Transform;
import Core.Log;
import Core.Input;
import Core.Time;
import Interpolator.AnimationCurve;

namespace SIByL::Editor
{
	struct CameraState
	{
		float yaw = 0;
		float pitch = 0;
		float roll = 0;

		float x = 0;
		float y = 0;
		float z = 0;

		auto setFromTransform(GFX::Transform const& transform) noexcept -> void
		{
			glm::vec3 eulerAngles = transform.getEulerAngles();
			glm::vec3 translation = transform.getTranslation();

			pitch = eulerAngles.x;
			yaw = eulerAngles.y;
			roll = eulerAngles.z;

			x = translation.x;
			y = translation.y;
			z = translation.z;
		}

		auto translate(glm::vec3 const& translation) noexcept -> void
		{
			glm::vec3 rotatedTranslation = glm::toMat3(glm::quat(glm::vec3{ pitch,yaw,roll }))* translation;
			
			x += rotatedTranslation.x;
			y += rotatedTranslation.y;
			z += rotatedTranslation.z;
		}

		auto lerpTowards(CameraState const& target, float positionLerpPct, float rotationLerpPct) noexcept -> void
		{
			yaw = std::lerp(yaw, target.yaw, rotationLerpPct);
			pitch = std::lerp(pitch, target.pitch, rotationLerpPct);
			roll = std::lerp(roll, target.roll, rotationLerpPct);

			x = std::lerp(x, target.x, positionLerpPct);
			y = std::lerp(y, target.y, positionLerpPct);
			z = std::lerp(z, target.z, positionLerpPct);
		}

		auto updateTransform(GFX::Transform& transform) noexcept -> void
		{
			transform.setEulerAngles(glm::vec3{ pitch,yaw,roll });
			transform.setTranslation(glm::vec3{ x,y,z });
		}
	};

	export struct SimpleCameraController
	{
		SimpleCameraController(IInput* input, Timer* timer) :input(input), timer(timer) {}

		float mouseSensitivityMultiplier = 0.01f;
		CameraState targetCameraState;
		CameraState interpolatingCameraState;
		float boost = 3.5f;
		float positionLerpTime = 0.2f;
		float mouseSensitivity = 60.0f;
		Interpolator::AnimationCurve mouseSensitivityCurve = { {0,0.5,0,5}, {1,2.5,0,0} };
		float rotationLerpTime = 0.01f;
		bool invertY = false;

		auto onEnable(GFX::Transform const& transform) noexcept -> void
		{
			targetCameraState.setFromTransform(transform);
			interpolatingCameraState.setFromTransform(transform);
		}

		auto getInputTranslationDirection() noexcept -> glm::vec3
		{
			glm::vec3 direction{ 0.0f,0.0f,0.0f };
			if (input->isKeyPressed(SIByL_KEY_W))
			{
				direction += glm::vec3(0, 0, 1); // forward
			}
			if (input->isKeyPressed(SIByL_KEY_S))
			{
				direction += glm::vec3(0, 0, -1); // back
			}
			if (input->isKeyPressed(SIByL_KEY_A))
			{
				direction += glm::vec3(-1, 0, 0); // left
			}
			if (input->isKeyPressed(SIByL_KEY_D))
			{
				direction += glm::vec3(1, 0, 0); // right
			}
			if (input->isKeyPressed(SIByL_KEY_Q))
			{
				direction += glm::vec3(0, -1, 0); // down
			}
			if (input->isKeyPressed(SIByL_KEY_E))
			{
				direction += glm::vec3(0, 1, 0); // up
			}
			return direction;
		}

		auto bindTransform(GFX::Transform* transform) noexcept -> void
		{
			interpolatingCameraState.setFromTransform(*transform);
			this->transform = transform;
		}

		auto onUpdate() noexcept -> void
		{
			// rotation
			static bool justPressedMouse = true;
			static float last_x = 0;
			static float last_y = 0;
			if (input->isMouseButtonPressed(SIByL_MOUSE_BUTTON_2))
			{
				input->disableCursor();
				float x = input->getMouseX();
				float y = input->getMouseY();
				if (justPressedMouse)
				{
					last_x = x;
					last_y = y;
					justPressedMouse = false;
				}
				else
				{
					glm::vec2 mouseMovement = glm::vec2(x - last_x, y - last_y) * 0.0005f * mouseSensitivityMultiplier * mouseSensitivity;
					if (invertY)
						mouseMovement.y = -mouseMovement.y;
					last_x = x;
					last_y = y;

					float mouseSensitivityFactor = mouseSensitivityCurve.evaluate(mouseMovement.length());

					targetCameraState.yaw += mouseMovement.x * mouseSensitivityFactor;
					targetCameraState.pitch += mouseMovement.y * mouseSensitivityFactor;
				}
			}
			else if (!justPressedMouse)
			{
				input->enableCursor();
				justPressedMouse = true;
			}

			// translation
			glm::vec3 translation = getInputTranslationDirection();
			translation *= timer->getMsPF() * 0.001;

			// speed up movement when shift key held
			if (input->isKeyPressed(SIByL_KEY_LEFT_SHIFT))
			{
				translation *= 10.0f;
			}

			// modify movement by a boost factor ( defined in Inspector and modified in play mode through the mouse scroll wheel)
			float y = input->getMouseScrollY();
			SE_CORE_DEBUG("Boost {0}, {1}", boost, y);
			translation *= powf(2.0f, boost);

			targetCameraState.translate(translation);

			// Framerate-independent interpolation
			// calculate the lerp amount, such that we get 99% of the way to our target in the specified time
			float positionLerpPct = 1.f - expf(log(1.f - 0.99f) / positionLerpTime * timer->getMsPF() * 0.001);
			float rotationLerpPct = 1.f - expf(log(1.f - 0.99f) / rotationLerpTime * timer->getMsPF() * 0.001);
			interpolatingCameraState.lerpTowards(targetCameraState, positionLerpPct, rotationLerpPct);

			if (transform != nullptr) interpolatingCameraState.updateTransform(*transform);
		}

	private:
		GFX::Transform* transform = nullptr;
		IInput* input;
		Timer* timer;
	};


}