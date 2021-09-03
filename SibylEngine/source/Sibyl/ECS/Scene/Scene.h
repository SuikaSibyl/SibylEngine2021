#pragma once

#include "entt.hpp"

namespace SIByL
{
	class Entity;
	class Scene
	{
	public:
		friend class Entity;

	public:
		Scene();
		~Scene();

		void OnUpdate();

		Entity CreateEntity(const std::string& name = "New Entity");

	private:
		entt::registry m_Registry;
		friend class SceneHierarchyPanel;
	};
}