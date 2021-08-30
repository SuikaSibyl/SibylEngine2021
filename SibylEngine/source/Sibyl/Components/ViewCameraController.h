#pragma once

#include "Sibyl/Graphic/Core/Camera.h"
#include "Sibyl/Events/Event.h"
#include "Sibyl/Events/MouseEvent.h"
#include "Sibyl/Events/ApplicationEvent.h"

namespace SIByL
{
	class ViewCameraController
	{
	public:
		ViewCameraController() = default;
		ViewCameraController(Ref<Camera> camera);

		void OnUpdate();

		void OnEvent(Event& e);

		
	private:
		struct CameraState
		{
			float yaw;
			float pitch;
			float roll;
			float x;
			float y;
			float z;

			void SetFromTransform(Ref<Transform> t);
			void Translate(glm::vec3 translation);
			void LerpTowards(CameraState target, float positionLerpPct, float rotationLerpPct);
			void UpdateTransform(Ref<Transform> t);
			Ref<Transform> AsTransform();
		};

	private:
		bool OnMouseScrolled(MouseScrolledEvent& e);
		bool OnWindowResized(WindowResizeEvent& e);

	private:
		void CheckCameraBinded();
		float RotationCurveEvalue(float input)
		{
			if (input > 1) return 2.5;
			else if (input < 0)return 0;
			else return -2.5 * input * input + 5 * input;
		}

	private:
		Ref<Camera> m_Camera = nullptr;
		float m_Boost = 3.5f;
		float m_Speed = 1.f;
		float m_RotateSpeed = .4f;
		float m_PositionLerpTime = 0.2f;
		float m_RotationLerpTime = 0.01f;

		CameraState m_TargetState;
		CameraState m_InterpolatingState;
	};
}