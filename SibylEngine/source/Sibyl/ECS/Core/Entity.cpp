#include "SIByLpch.h"
#include "Entity.h"
#include "Sibyl/ECS/UniqueID/UniqueID.h"

#include "Sibyl/ECS/Components/Components.h"

namespace SIByL
{
	Entity::Entity(const Entity& entt)
		:m_EntityHandle(entt.m_EntityHandle), m_Scene(entt.m_Scene)
	{

	}

	Entity::Entity(entt::entity handle, Scene* scene)
		:m_EntityHandle(handle), m_Scene(scene)
	{

	}

	Entity::Entity(const uint64_t& uid, Scene* scene)
		:m_EntityHandle(UniqueID::GetEid(uid)), m_Scene(scene)
	{

	}

	uint64_t Entity::GetUid() { return GetComponent<TransformComponent>().GetUid(); }
	void Entity::SetUid(const uint64_t& u) { GetComponent<TransformComponent>().SetUid(u); }
}