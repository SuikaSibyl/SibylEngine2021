#pragma once

namespace SIByL
{
	class Scene;
	class Camera;
	class FrameBuffer;

	class ViewportPanel
	{
	public:
		ViewportPanel() = default;

		void SetFrameBuffer(Ref<FrameBuffer> framebuffer, unsigned index = 0);
		void SetCamera(Ref<Camera> camera);
		const glm::vec2& GetViewportSize();
		bool IsViewportFocusd();
		bool IsViewportHovered();

		void OnImGuiRender();
		bool OnKeyPressed(KeyPressedEvent& e);

	protected:
		void ImGuiDrawMenu();
		void DrawFrameBufferItem(const std::string& name, Ref<FrameBuffer> frameBuffer);

	protected:
		bool m_ViewportFocused;
		bool m_ViewportHoverd;
		glm::vec2 m_ViewportSize;

		int GizmoType = -1;

		Ref<FrameBuffer> m_FrameBuffer;
		int m_FrameBufferIndex = 0;
		Ref<Camera> m_Camera;
	};

}