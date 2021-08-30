#include "SIByLpch.h"
#include "ViewCameraController.h"
#include "Sibyl/Core/Input.h"
#include "Sibyl/Core/KeyCodes.h"
#include "Sibyl/Core/MouseButtonCodes.h"
#include "Sibyl/Core/Application.h"

namespace SIByL
{
	ViewCameraController::ViewCameraController(Ref<Camera> camera)
		:m_Camera(camera)
	{
		m_TargetState.SetFromTransform(camera->GetTransform());
		m_InterpolatingState.SetFromTransform(camera->GetTransform());
	}

	float Lerp(float a, float b, float t)
	{
		return a * (1 - t) + b * t;
	}

	///////////////////////////////////////////////////////////////////////////////
	//                               Camera State                                //
	///////////////////////////////////////////////////////////////////////////////

	void ViewCameraController::CameraState::SetFromTransform(Ref<Transform> t)
	{
		pitch = t->eulerAngles.x;
		yaw = t->eulerAngles.y;
		roll = t->eulerAngles.z;
		x = t->position.x;
		y = t->position.y;
		z = t->position.z;
	}

	void ViewCameraController::CameraState::Translate(glm::vec3 translation)
	{
		x += translation.x;
		y += translation.y;
		z += translation.z;
	}

	void ViewCameraController::CameraState::LerpTowards(CameraState target, float positionLerpPct, float rotationLerpPct)
	{
		yaw = Lerp(yaw, target.yaw, rotationLerpPct);
		pitch = Lerp(pitch, target.pitch, rotationLerpPct);
		roll = Lerp(roll, target.roll, rotationLerpPct);

		x = Lerp(x, target.x, positionLerpPct);
		y = Lerp(y, target.y, positionLerpPct);
		z = Lerp(z, target.z, positionLerpPct);
	}

	void ViewCameraController::CameraState::UpdateTransform(Ref<Transform> t)
	{
		t->eulerAngles = glm::vec3(pitch, yaw, roll);
		t->position = glm::vec3(x, y, z);
	}

	Ref<Transform> ViewCameraController::CameraState::AsTransform()
	{
		Ref<Transform> t = std::make_shared<Transform>();
		t->eulerAngles = glm::vec3(pitch, yaw, roll);
		t->position = glm::vec3(x, y, z);
		return t;
	}

	///////////////////////////////////////////////////////////////////////////////
	//                             Camera Controller                             //
	///////////////////////////////////////////////////////////////////////////////

	void ViewCameraController::OnUpdate()
	{
		// Rotation Data
		static float prev_x = 0, prev_y = 0;
		static bool prev_IsMoving = false;
		
		// Check whether any camera is Binded
		CheckCameraBinded();
		float deltaTime = Application::Get().GetFrameTimer()->DeltaTime();

		// Right button rotation handling
		bool isMoving = false;
		float multiplier = Input::IsKeyPressed(SIByL_KEY_LEFT_SHIFT) ? m_Boost : 1.;
		if (Input::IsMouseButtonPressed(SIByL_MOUSE_BUTTON_RIGHT))
		{
			isMoving = true;
			if (prev_IsMoving == false)
			{
				prev_IsMoving = true;
				std::pair<float, float> pos = Input::GetMousePosition();
				prev_x = pos.first; prev_y = pos.second;
			}
			else
			{
				std::pair<float, float> pos = Input::GetMousePosition();
				float diffx = pos.first - prev_x;
				float diffy = pos.second - prev_y;
				glm::vec3 currRotation = m_Camera->GetRotation();

				float mouseSensitivityFactor = RotationCurveEvalue(glm::vec2(diffx, diffy).length() * 1. / 10);

				m_TargetState.pitch += -m_RotateSpeed * 400 * deltaTime * diffy * mouseSensitivityFactor;
				m_TargetState.yaw += +m_RotateSpeed * 400 * deltaTime * diffx * mouseSensitivityFactor;
				prev_x = pos.first; prev_y = pos.second;
			}

			if (Input::IsKeyPressed(SIByL_KEY_W))
			{
				//glm::vec3 currPosition = m_Camera->GetPosition();
				//m_Camera->SetPosition(currPosition + m_Camera->MoveForward({ 0, 0, m_Speed * 2 * deltaTime }));
				m_TargetState.Translate(m_Camera->MoveForward({ 0, 0, m_Speed * 2 * deltaTime * multiplier }));
			}
			if (Input::IsKeyPressed(SIByL_KEY_A))
			{
				//glm::vec3 currPosition = m_Camera->GetPosition();
				//m_Camera->SetPosition(currPosition + m_Camera->MoveForward({ -m_Speed * 2 * deltaTime, 0, 0 }));
				m_TargetState.Translate(m_Camera->MoveForward({ -m_Speed * 2 * deltaTime * multiplier, 0, 0 }));

			}
			if (Input::IsKeyPressed(SIByL_KEY_S))
			{
				//glm::vec3 currPosition = m_Camera->GetPosition();
				//m_Camera->SetPosition(currPosition + m_Camera->MoveForward({ 0, 0, -m_Speed * 2 * deltaTime }));
				m_TargetState.Translate(m_Camera->MoveForward({ 0, 0, -m_Speed * 2 * deltaTime * multiplier }));
			}
			if (Input::IsKeyPressed(SIByL_KEY_D))
			{
				//glm::vec3 currPosition = m_Camera->GetPosition();
				//m_Camera->SetPosition(currPosition + m_Camera->MoveForward({ +m_Speed * 2 * deltaTime, 0, 0 }));
				m_TargetState.Translate(m_Camera->MoveForward({ +m_Speed * 2 * deltaTime * multiplier, 0, 0 }));
			}
			if (Input::IsKeyPressed(SIByL_KEY_Q))
			{
				//glm::vec3 currPosition = m_Camera->GetPosition();
				//m_Camera->SetPosition(currPosition + m_Camera->MoveForward({ 0, -m_Speed * 2 * deltaTime, 0 }));
				m_TargetState.Translate(m_Camera->MoveForward({ 0, -m_Speed * 2 * deltaTime * multiplier, 0 }));
			}
			if (Input::IsKeyPressed(SIByL_KEY_E))
			{
				//glm::vec3 currPosition = m_Camera->GetPosition();
				//m_Camera->SetPosition(currPosition + m_Camera->MoveForward({ 0, +m_Speed * 2 *deltaTime, 0 }));
				m_TargetState.Translate(m_Camera->MoveForward({ 0, +m_Speed * 2 * deltaTime * multiplier, 0 }));
			}
		}
		else
		{
			prev_IsMoving = false;
		}

		float positionLerpPct = 1.f - exp((log(1.f - 0.99f) / m_PositionLerpTime) * deltaTime);
		float rotationLerpPct = 1.f - exp((log(1.f - 0.99f) / m_RotationLerpTime) * deltaTime);
		m_InterpolatingState.LerpTowards(m_TargetState, positionLerpPct, rotationLerpPct);
		m_Camera->SetTransform(m_InterpolatingState.AsTransform());
	}

	void ViewCameraController::OnEvent(Event& e)
	{
		if (e.GetEventType() == SIByL::EventType::WindowResize)
		{
			auto& re = (SIByL::WindowResizeEvent&)e;
			m_Camera->Resize(re.GetWidth(), re.GetHeight());
		}
	}

	bool ViewCameraController::OnMouseScrolled(MouseScrolledEvent& e)
	{
		return false;
	}

	bool ViewCameraController::OnWindowResized(WindowResizeEvent& e)
	{
		return false;
	}

	void ViewCameraController::CheckCameraBinded()
	{
		SIByL_CORE_ASSERT(m_Camera, "Camera Not Binded to Controller yet");
	}
}