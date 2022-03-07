module;
#include <cstdint>
#include <tuple>
#include <utility>
#include <functional>
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

		//template<typename Entity, typename... Exclude, typename... Component>
		//using basic_view = entt::basic_view<Entity, entt::exclude_t<Exclude...>, Component...>;

		template<typename... Component, typename... Exclude>
		entt::basic_view<entt::entity, entt::exclude_t<Exclude...>, Component...> view(entt::exclude_t<Exclude...> = {}) {
			return registry.view<Component...>();
		}

		template<class... Component, class... ComponentRef>
		void traverse(std::function<void(ComponentRef...)> fn) {
			auto components_view = view<Component...>();
			for (auto entity : components_view)
			{
				if constexpr (sizeof...(Component) == 1) {
					auto& component = components_view.get<Component...>(entity);
					fn(component);
				}
				else {
					std::tuple<ComponentRef...> components = components_view.get<Component...>(entity);
					fn(std::get<ComponentRef>(std::forward<std::tuple<ComponentRef...>>(components))...);
				}
			}
		}

		template<typename T>
		void OnComponentAdded(Entity entity, T& component) {}

		template<typename T>
		void OnComponentRemoved(Entity entity, T& component) {}

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
		T& addComponent(Args&&... args)
		{
			SE_CORE_ASSERT(!hasComponent<T>(), "ECS :: Entity already has component!");
			T& component = context->registry.emplace<T>(entityHandle, std::forward<Args>(args)...);
			context->OnComponentAdded<T>(*this, component);
			return component;
		}

		template<typename T>
		T& getComponent()
		{
			SE_CORE_ASSERT(hasComponent<T>(), "ECS :: Entity does not have component!");
			return context->registry.get<T>(entityHandle);
		}

		template<typename T>
		bool hasComponent()
		{
			return context->registry.all_of<T>(entityHandle);
		}

		template<typename T>
		bool removeComponent()
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