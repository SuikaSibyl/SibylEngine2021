#include "SIByLpch.h"
#include "Scene.h"

#include "glm/glm.hpp"

#include "Sibyl/Graphic/AbstractAPI/Core/Top/DrawItem.h"
#include "Sibyl/ECS/Core/Entity.h"
#include "Sibyl/ECS/Components/Components.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Camera.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h"
#include "Sibyl/Graphic/Core/Geometry/TriangleMesh.h"
#include "Sibyl/Graphic/Core/Lighting/LightManager.h"
#include "Sibyl/ECS/UniqueID/UniqueID.h"

#define DEFAULT_ONCOMPONENTADD(T) template<> void Scene::OnComponentAdded<T>(Entity entity, T& component) {}
#define DEFAULT_ONCOMPONENTREMOVE(T) template<> void Scene::OnComponentRemoved<T>(Entity entity, T& component) {}

namespace SIByL
{
	Scene::Scene()
	{

	}

	Scene::~Scene()
	{

	}

	void Scene::OnUpdate()
	{
		// Update Light
		{
			auto view = m_Registry.view<TransformComponent, LightComponent>();
			for (auto entity : view)
			{
				auto& [transform, light] =
					view.get<TransformComponent, LightComponent>(entity);

				glm::mat4 transformMatrix = transform.GetTransform();
				glm::vec3 dir = transformMatrix * glm::vec4(0, 0, 1, 0);
				light.m_Direction = glm::normalize(dir);
			}
		}
		LightManager::OnUpdate();

		// Update Draw Items Pool
		{
			DIPool.Reset();
			auto view = m_Registry.view<TransformComponent, MeshFilterComponent, MeshRendererComponent>();
			for (auto entity : view)
			{
				auto& [transform, meshFilter, meshRenderer] = 
					view.get<TransformComponent, MeshFilterComponent, MeshRendererComponent>(entity);

				meshFilter.PerObjectBuffer->SetMatrix4x4("Model", transform.GetTransform());

				int submeshIndex = 0;

				for (auto& submesh : *(meshFilter.Mesh))
				{
					Ref<DrawItem> item = DIPool.Request();
					item->m_Mesh = meshFilter.Mesh;
					item->m_SubMesh = &submesh;
					item->m_ConstantsBuffer = meshFilter.PerObjectBuffer;
					item->m_Material = meshRenderer.Materials[submeshIndex++];
					DIPool.Push(item);
				}
			}
		}
	}

	Entity Scene::CreateEntity(const std::string& name)
	{
		Entity entity { m_Registry.create(),this };
		entity.AddComponent<TagComponent>(name);
		entity.AddComponent<TransformComponent>();
		uint64_t uid = UniqueID::RequestUniqueID();
		entity.SetUid(uid);
		UniqueID::InsertUidEidPair(uid, entity);
		return entity;
	}

	void Scene::DestroyEntity(Entity entity)
	{
		m_Registry.destroy(entity);
	}

	template<typename T>
	void Scene::OnComponentAdded(Entity entity, T& component)
	{

	}

	template<>
	void Scene::OnComponentAdded<TransformComponent>(Entity entity, TransformComponent& component)
	{
		component.scene = entity;
	}

	DEFAULT_ONCOMPONENTADD(SpriteRendererComponent);
	DEFAULT_ONCOMPONENTADD(MeshFilterComponent);
	DEFAULT_ONCOMPONENTADD(MeshRendererComponent);

	template<>
	void Scene::OnComponentAdded<LightComponent>(Entity entity, LightComponent& component)
	{
		LightManager::AddLight(&component);
	}

	template<typename T>
	void Scene::OnComponentRemoved(Entity entity, T& component)
	{

	}
	
	DEFAULT_ONCOMPONENTREMOVE(TransformComponent);
	DEFAULT_ONCOMPONENTREMOVE(SpriteRendererComponent);
	DEFAULT_ONCOMPONENTREMOVE(MeshFilterComponent);
	DEFAULT_ONCOMPONENTREMOVE(MeshRendererComponent);

	template<>
	void Scene::OnComponentRemoved<LightComponent>(Entity entity, LightComponent& component)
	{
		LightManager::RemoveLight(&component);
	}

}