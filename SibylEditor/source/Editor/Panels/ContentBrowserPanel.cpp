#include "SIByLpch.h"
#include "ContentBrowserPanel.h"

#include <imgui.h>
#include <filesystem>

namespace SIByL
{
	constexpr const char* s_AssetsDirectory = "../Assets";

	ContentBrowserPanel::ContentBrowserPanel()
		:m_CurrentDirectory(s_AssetsDirectory)
	{

	}

	void ContentBrowserPanel::OnImGuiRender()
	{
		ImGui::Begin("Content Browser");

		for (auto& p : std::filesystem::directory_iterator(s_AssetsDirectory))
		{
			std::string path = p.path().string();
			if (p.is_directory())
			{
				if (ImGui::Button(path.c_str()))
				{

				}
			}
		}

		ImGui::End();
	}

}