#pragma once

#include "entt.hpp"
#include "DrawItemPool.h"

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
		void DestroyEntity(Entity entity);
		DrawItemPool& GetDrawItems() { return DIPool; };

	private:
		entt::registry m_Registry;
		friend class SceneHierarchyPanel;

		DrawItemPool DIPool;

		template<typename T>
		void OnComponentAdded(Entity entity, T& component);

		friend class SceneSerializer;
	};
}