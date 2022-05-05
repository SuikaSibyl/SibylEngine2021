module;
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <imgui.h>
#include <imgui_internal.h>
export module Editor.Statistics;
import Core.Time;
import Editor.Widget;

namespace SIByL::Editor
{
	export struct Statistics :public Widget
	{
		Statistics(Timer* timer);

		virtual auto onDrawGui() noexcept -> void override;

		float portalDrawcallTime = 0;
		float portalSoftDrawTime = 0;

	private:
		std::string fps;
		std::string MsPF;
		std::string portalDrawcallTimeStr;
		std::string portalportalSoftDrawTimeStr;
		float timePeriod = 0;
		float timeThereshold = 200;
		Timer* timer;
	};

	Statistics::Statistics(Timer* timer)
		:timer(timer)
	{}

	auto Statistics::onDrawGui() noexcept -> void
	{
		ImGui::Begin("Statistics", 0);

		timePeriod += (float)timer->getMsPF();
		if (timePeriod > timeThereshold)
		{
			fps = "FPS : " + std::to_string(timer->getFPS());
			MsPF = "MsPF : " + std::to_string(timer->getMsPF());
			portalDrawcallTimeStr = "Portal Drawcall Time (ms): " + std::to_string(portalDrawcallTime);
			portalportalSoftDrawTimeStr = "Portal SoftDraw Time (ms): " + std::to_string(portalSoftDrawTime);
			timePeriod = timePeriod - timeThereshold * ((int)(timePeriod / timeThereshold));
			if (timePeriod > timeThereshold) timePeriod = 0;
		}

		ImGui::Text(fps.c_str());
		ImGui::Text(MsPF.c_str());
		ImGui::Text(portalDrawcallTimeStr.c_str());
		ImGui::Text(portalportalSoftDrawTimeStr.c_str());

		ImGui::End();
	}

}