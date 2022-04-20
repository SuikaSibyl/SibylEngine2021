module;
#include <cstdint>
#include <imgui.h>
#include <imguizmo/ImGuizmo.h>
#include <glm/glm.hpp>
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
import GFX.Scene;
import Editor.Widget;
import Editor.ImImage;
import Editor.CameraController;
import Math.LinearAlgebra;
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
		auto getNeedResize() noexcept -> bool { return needResize; }
		GFX::Transform cameraTransform;

	private:
		bool needResize = false;
		IInput* input;
		int gizmozType = 0;
		auto handleTransformGizmo() noexcept -> void;
		SimpleCameraController cameraController;
	};

	Viewport::Viewport(WindowLayer* window_layer, Timer* timer)
		:input(window_layer->getWindow()->getInput()),
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

		//m_ViewportFocused = ImGui::IsWindowFocused();
		//m_ViewportHoverd = ImGui::IsWindowHovered();
		//Application::get().GetImGuiLayer()->SetBlockEvents(!m_ViewportFocused && !m_ViewportHoverd);

		ImVec2 viewport_panel_size = ImGui::GetContentRegionAvail();
		if ((viewportPanelSize.x != viewport_panel_size.x) || (viewportPanelSize.y != viewport_panel_size.y))
		{
			// Viewport Change Size
			viewportPanelSize = viewport_panel_size;
			needResize = true;
		}
		else if (needResize)
			needResize = false;

		if (bindedImage)
		{
			ImGui::Image(
				bindedImage->getImImage()->getImTextureID(), 
				{ (float)getWidth(),(float)getHeight() },
				{ 0,0 }, { 1, 1 });
		}

		handleTransformGizmo();

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
			glm::mat4 transform = tc.invalidTransform();

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
				glm::vec3 translation, rotation, scale;
				Math::decomposeTransform(transform, translation, rotation, scale);
				glm::vec3 deltaRotation = rotation - tc.getEulerAngles();
				tc.setTranslation(translation);
				tc.setScale(scale);
				tc.setEulerAngles(deltaRotation);
				transform = tc.invalidTransform();
			}
		}
	}

}