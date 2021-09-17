#pragma once

namespace SIByL
{
	class Scene;
	class Entity;

	class SceneHierarchyPanel
	{
	public:
		SceneHierarchyPanel() {}
		SceneHierarchyPanel(const Ref<Scene>& scene);

		void SetContext(const Ref<Scene>& scene);
		
		void OnImGuiRender();

		Entity GetSelectedEntity() const;

	private:
		void DrawEntityNode(Entity entity);
		void DrawComponents(Entity entity);

	private:
		Ref<Scene> m_Context;
		friend class Scene;
	};
}