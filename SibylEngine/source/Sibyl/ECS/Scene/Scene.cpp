#include "SIByLpch.h"
#include "Scene.h"

#include "glm/glm.hpp"

#include "Sibyl/Graphic/AbstractAPI/Core/Top/DrawItem.h"
#include "Sibyl/ECS/Core/Entity.h"
#include "Sibyl/ECS/Components/Components.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Camera.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h"
#include "Sibyl/Graphic/Core/Geometry/TriangleMesh.h"

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
		{
			auto group = m_Registry.group<TransformComponent, CameraComponent>();
			for (auto entity : group)
			{
				auto& [transform, camera] = group.get<TransformComponent, CameraComponent>(entity);

				if (camera.Primary)
				{

				}
			}
		}

		// Update Draw Items Pool

		{
			DIPool.Reset();
			auto view = m_Registry.view<TransformComponent, MeshFilterComponent, MeshRendererComponent>();
			for (auto entity : view)
			{
				auto& [transform, meshFilter, meshRenderer] = 
					view.get<TransformComponent, MeshFilterComponent, MeshRendererComponent>(entity);

				meshFilter.PerObjectBuffer->SetMatrix4x4("Model", transform.GetTransform());

				if (meshFilter.Mesh == nullptr)
				{
					SIByL::VertexBufferLayout layout =
					{
						{SIByL::ShaderDataType::Float3, "POSITION"},
						{SIByL::ShaderDataType::Float2, "TEXCOORD"},
					};

					//const wchar_t* path = (const wchar_t*)payload->Data;
					//std::filesystem::path texturePath = path;
					SIByL::MeshLoader meshLoader(meshFilter.Path, layout);
					meshFilter.Mesh = meshLoader.GetTriangleMesh();

				}

				for (auto& submesh : *(meshFilter.Mesh))
				{
					Ref<DrawItem> item = DIPool.Request();
					item->m_Mesh = meshFilter.Mesh;
					item->m_SubMesh = &submesh;
					item->m_ConstantsBuffer = meshFilter.PerObjectBuffer;
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
	void Scene::OnComponentAdded<SpriteRendererComponent>(Entity entity, SpriteRendererComponent& component)
	{

	}

	template<>
	void Scene::OnComponentAdded<MeshFilterComponent>(Entity entity, MeshFilterComponent& component)
	{

	}

	template<>
	void Scene::OnComponentAdded<MeshRendererComponent>(Entity entity, MeshRendererComponent& component)
	{

	}
}