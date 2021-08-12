#pragma once

#include "Sibyl/Core/Layer.h"
#include "Sibyl/Events/MouseEvent.h"
#include "Sibyl/Events/KeyEvent.h"
#include "Sibyl/Events/ApplicationEvent.h"

namespace SIByL
{
	class SIByL_API ImGuiLayer :public Layer
	{
	public:
		ImGuiLayer();
		~ImGuiLayer();

		void OnAttach();
		void OnDetach();

		void OnUpdate();
		void OnEvent(Event& event);

		static ImGuiLayer* Create();

	protected:
		virtual void PlatformInit() {};
		virtual void NewFrameBegin() {};
		virtual void NewFrameEnd() {};
		virtual void PlatformDestroy() {};

	private:
		bool OnMouseButtonPressedEvent(MouseButtonPressedEvent& e);
		bool OnMouseMovedEvent(MouseMovedEvent& e);
		bool OnMouseButtonReleasedEvent(MouseButtonReleasedEvent& e);
		bool OnMouseScrolledEvent(MouseScrolledEvent& e);
		bool OnKeyPressedEvent(KeyPressedEvent& e);
		bool OnKeyReleasedEvent(KeyReleasedEvent& e);
		bool OnKeyTypedEvent(KeyTypedEvent& e);
		bool OnWindowResizeEvent(WindowResizeEvent& e);
				

	private:
		float m_Time = 0.0f;

	};
}