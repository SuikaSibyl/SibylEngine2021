#include "SIByLpch.h"
#include "ImGuiUtility.h"
#include "Sibyl/Graphic/Core/Renderer/Renderer.h"

namespace ImGui
{
	void DrawImage(ImTextureID user_texture_id, const ImVec2& size)
	{
		switch (SIByL::Renderer::GetRaster())
		{
		case SIByL::RasterRenderer::OpenGL:
			ImGui::Image(user_texture_id, size,
				{ 0,1 }, { 1, 0 });
			break;
		case SIByL::RasterRenderer::DirectX12: 
			ImGui::Image(user_texture_id, size,
				{ 0,0 }, { 1, 1 });
			break;
		default: ; break;
		}
	}
}