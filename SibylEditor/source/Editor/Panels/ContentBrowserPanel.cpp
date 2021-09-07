#include "SIByLpch.h"
#include "ContentBrowserPanel.h"

#include <imgui.h>
#include <filesystem>

namespace SIByL
{
	constexpr const char* s_AssetsDirectory = "../Assets";

	void ContentBrowserPanel::OnImGuiRender()
	{
		ImGui::Begin("Content Browser");

		for (auto& p : std::filesystem::directory_iterator(s_AssetsDirectory))
		{
			std::string path = p.path().string();
			ImGui::Text("%s", path.c_str());
		}

		ImGui::End();
	}

}