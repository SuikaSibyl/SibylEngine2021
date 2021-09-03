#pragma once

#include <functional>

namespace SIByL
{
	class ScriptableEntity
	{

	};

	struct NativeScriptComponent
	{
		ScriptableEntity* Instance = nullptr;

		std::function<void()> InstantiateFunction;
		std::function<void()> DestroyInstanceFunction;

		std::function<void(ScriptableEntity* instance)> OnCreateFunction;
		std::function<void(ScriptableEntity* instance)> OnUpdateFunction;
		std::function<void(ScriptableEntity* instance)> OnDestroyFunction;


		template<typename T>
		void Bind()
		{
			InstantiateFunction = [&Instance]() {Instance = new T(); };
			DestroyInstanceFunction = [&Instance]() {delte(T*)Instance; };

			OnCreateFunction = [](ScriptableEntity* instance) {(T*)instance->OnCreate(); }
			OnUpdateFunction = [](ScriptableEntity* instance) {(T*)instance->OnUpdate(); }
			OnDestroyFunction = [](ScriptableEntity* instance) {(T*)instance->OnDestroy(); }
		}
	};
}