#include "SIByLpch.h"
#include "Camera.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Top/Camera.h"

#include "Sibyl/ECS/Core/Entity.h"

namespace SIByL
{
	CameraComponent::CameraComponent(Ref<Camera> camera)
		:m_Camrea(camera)
	{

	}

	void TransformComponent::SetParent(const uint64_t& p)
	{
		if (parent != 0)
		{
			// Remove this node from its parent
			Entity parentEntity(parent, scene);
			parentEntity.GetComponent<TransformComponent>().RemoveChild(uid);
		}
		
		parent = p;
		if (parent != 0)
		{
			Entity parentEntity(parent, scene);
			parentEntity.GetComponent<TransformComponent>().AddChild(uid);
		}
	}

	void TransformComponent::AddChild(const uint64_t& c)
	{
		children.push_back(c);
	}

	void TransformComponent::RemoveChild(const uint64_t& c)
	{
		for (auto& iter = children.begin(); iter != children.end(); iter++)
		{
			if (*iter == c)
			{
				children.erase(iter);
				return;
			}
		}
	}

}