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
		DrawItemPool& GetDrawItems(std::string PassName) { return DIPool[PassName]; };

	private:
		entt::registry m_Registry;
		friend class SceneHierarchyPanel;

		std::unordered_map<std::string, DrawItemPool> DIPool;

		template<typename T>
		void OnComponentAdded(Entity entity, T& component);

		template<typename T>
		void OnComponentRemoved(Entity entity, T& component);

		friend class SceneSerializer;
	};
}