#include "SIByLpch.h"
#include "Entity.h"

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

}