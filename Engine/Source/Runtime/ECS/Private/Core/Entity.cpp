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
		entity.AddComponent<TagComponent>(name);
		//entity.AddComponent<TransformComponent>();
		//uint64_t uid = UniqueID::RequestUniqueID();
		//entity.SetUid(uid);
		//UniqueID::InsertUidEidPair(uid, entity);
		//return entity;
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
