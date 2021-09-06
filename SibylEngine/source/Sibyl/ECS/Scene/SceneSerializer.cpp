#include "SIByLpch.h"
#include "SceneSerializer.h"

#include "yaml-cpp/yaml.h"

#include "Sibyl/ECS/Core/Entity.h"
#include "Sibyl/ECS/Components/Components.h"

namespace SIByL
{
	SceneSerializer::SceneSerializer(const Ref<Scene>& scene)
		:m_Scene(scene)
	{

	}

	static void SerializeEntity(YAML::Emitter& out, Entity entity)
	{
		out << YAML::BeginMap;
		out << YAML::Key << "Entity" << YAML::Value << "124531623461";

		if (entity.HasComponent<TagComponent>())
		{
			out << YAML::Key << "TagComponent";
			out << YAML::BeginMap;

			auto& tag = entity.GetComponent<TagComponent>().Tag;
			out << YAML::Key << "Tag" << YAML::Value << tag;

			out << YAML::EndMap;
		}

		out << YAML::EndMap;
	}

	void SceneSerializer::Serialize(const std::string& filepath)
	{
		YAML::Emitter out;
		out << YAML::BeginMap;
		out << YAML::Key << "Scene" << YAML::Value << "Untitled";
		out << YAML::Key << "Entities" << YAML::Value << YAML::BeginSeq;

		m_Scene->m_Registry.each([&](auto entityID)
			{
				Entity entity = { entityID, m_Scene.get() };
				if (!entity)
					return;

				SerializeEntity(out, entity);
			});
	}

	void SceneSerializer::SerializeRuntime(const std::string& filepath)
	{
		// Not Implemented
		SIByL_CORE_ASSERT(false);
		return;
	}

	bool SceneSerializer::Deserialize(const std::string& filepath)
	{

	}

	bool SceneSerializer::DeserializeRuntime(const std::string& filepath)
	{
		// Not Implemented
		SIByL_CORE_ASSERT(false);
		return false;
	}

}