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
		virtual ~ImGuiLayer();

		void OnAttach();
		void OnDetach();
		virtual void OnDraw() override;
		virtual void OnDrawImGui() override;
		virtual void OnReleaseResource() override;
		virtual void OnDrawAdditionalWindowsImpl() = 0;
		static inline void OnDrawAdditionalWindows() { Main->OnDrawAdditionalWindowsImpl(); }
		void OnUpdate();
		void OnEvent(Event& event);

		static ImGuiLayer* Create();
		static inline ImGuiLayer* Get() { return Main; }
	protected:
		virtual void PlatformInit() {};
		virtual void NewFrameBegin() {};
		virtual void NewFrameEnd() {};
		virtual void PlatformDestroy() {};

	private:
		float m_Time = 0.0f;
		static ImGuiLayer* Main;
	};
}