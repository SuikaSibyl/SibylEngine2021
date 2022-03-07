module;
#include <cstdint>
#include "entt/entt.hpp"
module ECS.Entity;
import Core.Assert;
import ECS.TagComponent;

namespace SIByL::ECS
{
	auto Context::createEntity(std::string const& name) noexcept -> Entity
	{
		Entity entity{ registry.create(),this };
		entity.addComponent<TagComponent>(name);
		return entity;
	}

	auto Context::destroyEntity(Entity entity) noexcept -> void
	{
		registry.destroy(entity);
	}

	Entity::Entity(Entity const& entt)
		:entityHandle(entt.entityHandle), context(entt.context)
	{}

	Entity::Entity(entt::entity handle, Context* context)
		:entityHandle(handle), context(context)
	{}
}
