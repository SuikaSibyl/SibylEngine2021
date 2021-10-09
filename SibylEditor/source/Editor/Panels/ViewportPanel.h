#pragma once

namespace SIByL
{
	class Scene;

	class ViewportPanel
	{
	public:
		ViewportPanel() {}
		ViewportPanel(const Ref<Scene>& scene);

		void OnImGuiRender();

	private:
		Ref<Scene> m_Context;
		friend class Scene;
	};

}