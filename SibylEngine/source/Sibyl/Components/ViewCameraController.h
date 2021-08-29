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
		ViewCameraController();

		void OnUpdate();

		void OnEvent(Event& e);

	private:
		bool OnMouseScrolled(MouseScrolledEvent& e);
		bool OnWindowResized(WindowResizeEvent& e);

	private:
		Camera m_Camera;
	};
}