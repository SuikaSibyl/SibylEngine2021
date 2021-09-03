#pragma once

#include "Sibyl/ECS/Core/Entity.h"

namespace SIByL
{
	class ScriptableEntity
	{
	public:
		template<typename T>
		T& GetComponent()
		{
			return m_Entity.GetComponent<T>();
		}

	private:
		Entity m_Entity;
		friend class Scene;
	};
}