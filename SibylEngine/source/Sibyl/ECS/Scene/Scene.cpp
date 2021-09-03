#include "SIByLpch.h"
#include "Scene.h"

#include "glm/glm.hpp"

#include "Sibyl/ECS/Core/Entity.h"
#include "Sibyl/ECS/Components/Common/Tag.h"
#include "Sibyl/ECS/Components/Common/Transform.h"

namespace SIByL
{
	static void DoMath(const glm::mat4& transform)
	{

	}

	Scene::Scene()
	{
		struct TransformComponent
		{
			glm::mat4 Transform;

			TransformComponent() = default;
			TransformComponent(const TransformComponent&) = default;
			TransformComponent(const glm::mat4& transform)
				:Transform(transform) {}

			operator const glm::mat4& () const { return Transform; }
			operator glm::mat4& () { return Transform; }
		};

		//TransformComponent transform;
		//DoMath(transform);

		//entt::entity entity = m_Registry.create();
		//m_Registry.emplace<TransformComponent>(entity, glm::mat4(1.0f));

		////if (m_Registry.<TransformComponent>(entity))
		//TransformComponent& fetched = m_Registry.get<TransformComponent>(entity);
		
		//auto view = m_Registry.view<TransformComponent>();
		//for (auto entity : view)
		//{
		//	TransformComponent& transform = m_Registry.get<TransformComponent>(entity);

		//}

		//auto group = m_Registry.group<TransformComponent>(entt::get<MeshComponent>)
		//{

		//}
	}

	Scene::~Scene()
	{

	}

	void Scene::OnUpdate()
	{

	}

	Entity Scene::CreateEntity(const std::string& name)
	{
		Entity entity { m_Registry.create(),this };
		entity.AddComponent<TagComponent>(name);
		entity.AddComponent<TransformComponent>();
		return entity;
	}
}