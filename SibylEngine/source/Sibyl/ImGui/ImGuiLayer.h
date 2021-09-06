#pragma once

#include "Sibyl/Core/Layer.h"
#include "Sibyl/Core/Events/MouseEvent.h"
#include "Sibyl/Core/Events/KeyEvent.h"
#include "Sibyl/Core/Events/ApplicationEvent.h"

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

		void SetBlockEvents(bool block) { m_BlockEvents = block; }

	private:
		void SetDarkThemeColors();

	protected:
		virtual void PlatformInit() {};
		virtual void NewFrameBegin() {};
		virtual void DrawCall() {};
		virtual void PlatformDestroy() {};

	private:
		float m_Time = 0.0f;
		bool m_BlockEvents = false;
		static ImGuiLayer* Main;
	};
}