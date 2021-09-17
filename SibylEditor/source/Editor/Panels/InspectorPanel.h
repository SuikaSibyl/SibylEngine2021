#pragma once

namespace SIByL
{
	class Entity;
	class Material;

	class InspectorPanel
	{
	public:
		enum State
		{
			None,
			EntityComponents,
			MaterialEditor,
		};

	public:
		InspectorPanel();
		void OnImGuiRender();

		void SetSelectedEntity(const Entity& entity);
		void SetSelectedMaterial(Ref<Material> material);

		void DrawComponents(Entity entity);

		Entity m_SelectEntity = {};
		Ref<Material> m_SelectMaterial = nullptr;
		State m_State = State::None;
	};
} 