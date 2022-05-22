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
		float portalOneSweep4Time = 0;
		float grassTime = 0;

	private:
		std::string fps;
		std::string MsPF;
		std::string portalDrawcallTimeStr;
		std::string portalportalSoftDrawTimeStr;
		std::string portalOneSweep4TimeStr;
		std::string grassTimeStr;
		float timePeriod = 0;
		float timeThereshold = 200;
		Timer* timer;

		bool hasRecording = false;
		bool startRecording = false;
		float accumulateSortingTime = 0;
		float accumulateGrassTime = 0;
		float accumulatePortalTime = 0;
		unsigned int recordingSampleCount = 0;

		std::string avgPortalOneSweep4TimeStr;
		std::string avggrassTimeStr;
		std::string avgPortalRenderTimeStr;
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
			portalOneSweep4TimeStr = "OnesweepX4 Time (ms): " + std::to_string(portalOneSweep4Time);
			grassTimeStr = "Grass Time (ms): " + std::to_string(grassTime);
			timePeriod = timePeriod - timeThereshold * ((int)(timePeriod / timeThereshold));
			if (timePeriod > timeThereshold) timePeriod = 0;
		}
		if (startRecording == true)
		{
			recordingSampleCount += 1;
			accumulateSortingTime += portalOneSweep4Time;
			accumulateGrassTime += grassTime;
			accumulatePortalTime += portalDrawcallTime;
		}
		ImGui::Text(fps.c_str());
		ImGui::Text(MsPF.c_str());
		ImGui::Text(portalDrawcallTimeStr.c_str());
		ImGui::Text(portalportalSoftDrawTimeStr.c_str());
		ImGui::Text(portalOneSweep4TimeStr.c_str());
		ImGui::Text(grassTimeStr.c_str());

		if (ImGui::Button("Recording", ImVec2{ 30, 30 }))
		{
			if (startRecording == false)
			{
				startRecording = true;
				recordingSampleCount = 0;
				accumulateSortingTime = 0;
				accumulateGrassTime = 0;
				accumulatePortalTime = 0;
			}
			else
			{
				startRecording = false;
				hasRecording = true;
				accumulateSortingTime /= recordingSampleCount;
				accumulatePortalTime /= recordingSampleCount;
				accumulateGrassTime /= recordingSampleCount;
				avgPortalOneSweep4TimeStr = "Sorting Time Avg (ms): " + std::to_string(accumulateSortingTime);
				avgPortalRenderTimeStr = "Portal Time Avg (ms): " + std::to_string(accumulatePortalTime);
				avggrassTimeStr = "Grass Time Avg (ms): " + std::to_string(accumulateGrassTime);
			}

		}

		if (hasRecording)
		{
			ImGui::Text(avgPortalOneSweep4TimeStr.c_str());
			ImGui::Text(avgPortalRenderTimeStr.c_str());
			ImGui::Text(avggrassTimeStr.c_str());
		}

		ImGui::End();
	}

}