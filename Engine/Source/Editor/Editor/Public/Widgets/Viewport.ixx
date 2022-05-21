module;
#include <cstdint>
#include <vector>
#include <imgui.h>
#include <imguizmo/ImGuizmo.h>
#include <glm/glm.hpp>
#include <fstream>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "entt.hpp"
export module Editor.Viewport;
import Core.Window;
import Core.Time;
import Core.Input;
import ECS.Entity;
import ECS.UID;
import ECS.TagComponent;
import GFX.Transform;
import GFX.SceneTree;
import GFX.BoundingBox;
import GFX.Scene;
import Editor.Widget;
import Editor.ImImage;
import Editor.CameraController;
import Math.LinearAlgebra;
import Math.Geometry;
import Editor.RDGImImageManager;

namespace SIByL::Editor
{
	export struct Viewport :public Widget
	{
		Viewport(WindowLayer* window_layer, Timer* timer);

		virtual auto onDrawGui() noexcept -> void override;
		virtual auto onUpdate() noexcept -> void override;

		auto getWidth() noexcept -> uint32_t { return viewportPanelSize.x; }
		auto getHeight() noexcept -> uint32_t { return viewportPanelSize.y; }
		auto bindImImage(RDGImImage* image) noexcept -> void { bindedImage = image; }
		auto onKeyPressedEvent(KeyPressedEvent& e) -> bool;

		RDGImImage* bindedImage;
		ImVec2 viewportPanelSize = { 1280,720 };

		ECS::Entity selectedEntity = {};

		auto getCameraView() noexcept -> glm::mat4;
		auto getCameraProjection() noexcept -> glm::mat4;
		auto getNeedResize() noexcept -> bool { return needResize != 0; }
		GFX::Transform cameraTransform;

	private:
		bool firstResize = true;
		int needResize = 0;
		IInput* input;
		int gizmozType = 0;
		auto handleTransformGizmo() noexcept -> void;
		SimpleCameraController cameraController;
		WindowLayer* windowLayer;
	};

	Viewport::Viewport(WindowLayer* window_layer, Timer* timer)
		:input(window_layer->getWindow()->getInput()),
		windowLayer(window_layer),
		cameraController(input, timer)
	{ 
		cameraController.bindTransform(&cameraTransform);
	}

	auto Viewport::getCameraView() noexcept -> glm::mat4
	{
		return glm::lookAtLH(cameraTransform.getTranslation(), cameraTransform.getTranslation() + cameraTransform.getRotatedForward(), glm::vec3(0.0f, 1.0f, 0.0f));
	}

	auto Viewport::getCameraProjection() noexcept -> glm::mat4
	{
		return glm::perspectiveLH_NO(glm::radians(45.0f), (float)getWidth() / (float)getHeight(), 0.1f, 5000.0f);
	}

	auto Viewport::onDrawGui() noexcept -> void
	{
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2({ 0,0 }));
		ImGui::Begin("Viewport", 0, ImGuiWindowFlags_MenuBar);
		// Menu
		{
			ImGui::PushItemWidth(ImGui::GetFontSize() * -12);
			if (ImGui::BeginMenuBar())
			{
				// Menu - Save
				if (ImGui::BeginMenu("Save Setting"))
				{
					// Menu - File - Load
					if (ImGui::MenuItem("Save"))
					{
						std::string path = windowLayer->getWindow()->saveFile("");
						std::ofstream file1(path);
						auto transform = cameraTransform.getTranslation();
						auto euler = cameraTransform.getEulerAngles();
						file1 << transform.x << " " << transform.y << " " << transform.z << " ";
						file1 << euler.x << " " << euler.y << " " << euler.z;
						file1.close();
						//std::string path = windowLayer->getWindow()->openFile("");

						//hold_scene = MemNew<GFX::Scene>();
						//bindScene(hold_scene.get());
					}
					ImGui::EndMenu();
				}
				// Menu - Load
				if (ImGui::BeginMenu("Load Setting"))
				{
					// Menu - File - Load
					if (ImGui::MenuItem("Load"))
					{
						std::string path = windowLayer->getWindow()->openFile("");
						std::ifstream file1(path);
						auto transform = cameraTransform.getTranslation();
						auto euler = cameraTransform.getEulerAngles();
						file1 >> transform.x >> transform.y >> transform.z;
						file1 >> euler.x >> euler.y >> euler.z;
						file1.close();
						cameraTransform.setTranslation(transform);
						cameraTransform.setEulerAngles(euler);
						cameraController.targetCameraState.setFromTransform(cameraTransform);
					}
					ImGui::EndMenu();
				}
				ImGui::EndMenuBar();
			}
			ImGui::PopItemWidth();
		}
		//m_ViewportFocused = ImGui::IsWindowFocused();
		//m_ViewportHoverd = ImGui::IsWindowHovered();
		//Application::get().GetImGuiLayer()->SetBlockEvents(!m_ViewportFocused && !m_ViewportHoverd);

		ImVec2 viewport_panel_size = ImGui::GetContentRegionAvail();
		if ((viewportPanelSize.x != viewport_panel_size.x) || (viewportPanelSize.y != viewport_panel_size.y))
		{
			// Viewport Change Size
			viewportPanelSize = viewport_panel_size;
			//if (firstResize)
			//{
			//	needResize = 3;
			//	firstResize = false;
			//}
			//else
			needResize = 1;
		}
		else if (needResize)
			needResize--;

		ImGui::BeginChild("Test");
		ImVec2 p = ImGui::GetCursorScreenPos();
		if (bindedImage)
		{
			ImGui::Image(
				bindedImage->getImImage()->getImTextureID(), 
				{ (float)getWidth(),(float)getHeight() },
				{ 0,0 }, { 1, 1 });
		}
		handleTransformGizmo();
		ImGui::EndChild();

		ImGui::BeginChild("Test");
		if (selectedEntity)
		{
			// draw bounding box
			if (selectedEntity.hasComponent<GFX::BoundingBox>())
			{
				glm::mat4 cameraView = getCameraView();
				glm::mat4 cameraProj = getCameraProjection();
				auto& tc = selectedEntity.getComponent<GFX::Transform>();
				auto& bc = selectedEntity.getComponent<GFX::BoundingBox>();
				glm::mat4 transform = tc.getAccumulativeTransform();
				glm::vec3 offset = { transform[3][0],transform[3][1] ,transform[3][2] };
				std::vector<glm::vec3> points;
				for (int i = 0; i < 8; i++)
				{
					glm::vec3 position = bc.getNode(i);
					glm::vec4 projected = cameraProj * cameraView * glm::vec4(position + offset, 1);
					points.emplace_back((projected.x / abs(projected.w) + 1) * 0.5, (1 - projected.y / abs(projected.w)) * 0.5, projected.z / 5000.0f);
				}
				std::vector<int> bb_indices = GFX::getBBoxLinePairs();
				for (int i = 0; i < bb_indices.size(); i += 2)
				{
					glm::vec3 pos_0 = points[bb_indices[i]];
					glm::vec3 pos_1 = points[bb_indices[i + 1]];
					Math::clampLineUniformly(pos_0, pos_1);
					ImVec2 im_pos_0 = { p.x + pos_0.x * getWidth(),p.y + pos_0.y * getHeight() };
					ImVec2 im_pos_1 = { p.x + pos_1.x * getWidth(),p.y + pos_1.y * getHeight() };
					ImGui::GetWindowDrawList()->AddLine(im_pos_0, im_pos_1, IM_COL32(255, 0, 0, 100), 3.0f);
				}
			}
		}
		ImGui::EndChild();

		//unsigned int textureID = m_FrameBuffer->GetColorAttachment();
		//ImGui::DrawImage(
		//	(void*)m_FrameBuffer->GetColorAttachment(m_FrameBufferIndex), 
		//	ImVec2{viewportPanelSize.x,
		//	viewportPanelSize.y });
		
		ImGui::End();


		ImGui::PopStyleVar();
	}

	auto Viewport::onKeyPressedEvent(KeyPressedEvent& e) -> bool
	{
		if (e.getKeyCode() == input->decodeCodeEnum(SIByL_KEY_Q))
		{
			if (!input->isMouseButtonPressed(SIByL_MOUSE_BUTTON_RIGHT))
				gizmozType = 0;
		}
		else if (e.getKeyCode() == input->decodeCodeEnum(SIByL_KEY_W))
		{
			if (!input->isMouseButtonPressed(SIByL_MOUSE_BUTTON_RIGHT))
				gizmozType = ImGuizmo::OPERATION::TRANSLATE;
		}
		else if (e.getKeyCode() == input->decodeCodeEnum(SIByL_KEY_E))
		{
			if (!input->isMouseButtonPressed(SIByL_MOUSE_BUTTON_RIGHT))
				gizmozType = ImGuizmo::OPERATION::ROTATE;
		}
		else if (e.getKeyCode() == input->decodeCodeEnum(SIByL_KEY_R))
		{
			if (!input->isMouseButtonPressed(SIByL_MOUSE_BUTTON_RIGHT))
				gizmozType = ImGuizmo::OPERATION::SCALE;
		}
		return true;
	}

	auto Viewport::onUpdate() noexcept -> void
	{
		cameraController.onUpdate();
	}

	auto Viewport::handleTransformGizmo() noexcept -> void
	{
		if (selectedEntity && gizmozType != 0)
		{
			ImGuizmo::SetOrthographic(false);
			ImGuizmo::SetDrawlist();

			float windowWidth = (float)ImGui::GetWindowWidth();
			float windowHeight = (float)ImGui::GetWindowHeight();
			ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y, windowWidth, windowHeight);

			glm::mat4 cameraView = getCameraView();
			glm::mat4 cameraProj = getCameraProjection();

			// Entity transform
			auto& tc = selectedEntity.getComponent<GFX::Transform>();
			glm::mat4x4 transform = tc.getAccumulativeTransform();
			// Snapping 
			bool snap = input->isKeyPressed(SIByL_KEY_LEFT_CONTROL);
			float snapValue = 0.5f;
			if (gizmozType == ImGuizmo::OPERATION::ROTATE)
				snapValue = 45.0f;

			float snapValues[3] = { snapValue, snapValue ,snapValue };

			ImGuizmo::Manipulate(glm::value_ptr(cameraView), glm::value_ptr(cameraProj),
				ImGuizmo::OPERATION(gizmozType), ImGuizmo::LOCAL, glm::value_ptr(transform),
				nullptr, snap ? snapValues : nullptr);

			if (ImGuizmo::IsUsing())
			{
				transform = tc.getInversePrecursorTransform() * transform;
				glm::vec3 translation, rotation, scale;
				Math::decomposeTransform(transform, translation, rotation, scale);
				tc.setTranslation(translation);
				tc.setScale(scale);
				tc.setEulerAngles(rotation);
				transform = tc.invalidTransform();
				tc.reAccumule();
			}
		}
	}

}