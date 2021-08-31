#pragma once

#include "entt.hpp"

namespace SIByL
{
	class Scene
	{
	public:
		Scene();
		~Scene();

		void OnUpdate();

		entt::entity CreateEntity();

	private:
		entt::registry m_Registry;
	};
}