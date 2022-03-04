module;
#include <cstdint>
#include "entt/entt.hpp"
export module ECS.Entity;
import Core.Assert;


namespace SIByL::ECS
{
	class Entity;

	export class Context
	{
	public:
		auto createEntity(std::string const& name = "New Entity") noexcept -> Entity;
		auto destroyEntity(Entity entity) noexcept -> void;

		template<typename T>
		void OnComponentAdded(Entity entity, T& component) {}

		template<typename T>
		void OnComponentRemoved(Entity entity, T& component) {}

	private:
		friend class Entity;
		entt::registry registry;
	};
	
	export class Entity
	{
	public:
		Entity() = default;
		Entity(Entity const& entt);
		Entity(entt::entity handle, Context* context);

		template<typename T, typename ... Args>
		T& AddComponent(Args&&... args)
		{
			SE_CORE_ASSERT(!HasComponent<T>(), "ECS :: Entity already has component!");
			T& component = context->registry.emplace<T>(entityHandle, std::forward<Args>(args)...);
			context->OnComponentAdded<T>(*this, component);
			return component;
		}

		template<typename T>
		T& GetComponent()
		{
			SE_CORE_ASSERT(HasComponent<T>(), "ECS :: Entity does not have component!");
			return context->registry.get<T>(entityHandle);
		}

		template<typename T>
		bool HasComponent()
		{
			return context->registry.all_of<T>(entityHandle);
		}

		template<typename T>
		bool RemoveComponent()
		{
			SE_CORE_ASSERT(HasComponent<T>(), "ECS :: Entity does not have component!");
			context->OnComponentRemoved<T>(*this, GetComponent<T>());
			return context->registry.remove<T>(entityHandle);
		}

		// implicit casts
		operator entt::entity() const { return entityHandle; }
		operator bool() const { return entityHandle != entt::null; }
		operator uint32_t() const { return (uint32_t)entityHandle; }
		operator Context*() const { return context; }

		bool operator==(Entity const& other) const {
			return entityHandle == other.entityHandle && context == other.context;
		}

		bool operator!=(Entity const& other) const {
			return !(*this == other);
		}

	private:
		entt::entity entityHandle = entt::null;
		Context* context;
	};
}