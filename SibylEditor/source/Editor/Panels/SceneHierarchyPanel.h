#pragma once

namespace SIByL
{
	class Scene;
	class Entity;

	class SceneHierarchyPanel
	{
	public:
		SceneHierarchyPanel(const Ref<Scene>& scene);

		void SetContext(const Ref<Scene>& scene);
		
		void OnImGuiRender();

		Entity GetSelectedEntity() const { return m_SelectContext; }

	private:
		void DrawEntityNode(Entity entity);
		void DrawComponents(Entity entity);

	private:
		Ref<Scene> m_Context;
		Entity m_SelectContext;
		friend class Scene;
	};
}