#pragma once

#include "Sibyl/ImGui/ImGuiLayer.h"

namespace SIByL
{
	class ImGuiLayerOpenGL :public ImGuiLayer
	{
	protected:
		void PlatformInit() override;
		void NewFrameBegin() override;
		void NewFrameEnd() override;
	};
}