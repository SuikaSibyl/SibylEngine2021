#pragma once

#include "Sibyl/ImGui/ImGuiLayer.h"

namespace SIByL
{
	class ImGuiLayerOpenGL :public ImGuiLayer
	{
	protected:
		virtual void PlatformInit() override;
		virtual void NewFrameBegin() override;
		virtual void NewFrameEnd() override;
		virtual void PlatformDestroy() override;

	};
}