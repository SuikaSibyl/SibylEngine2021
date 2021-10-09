#include "SIByLpch.h"
#include "ModuleManager.h"

#include <NetworkModule/include/NetworkModule.h>
#include "ShaderModule/ShaderModule.h"

namespace SIByL
{
	void ModuleManager::Init()
	{
		ShaderModule::Init();
		SIByLNetwork::NetworkModule::Init();
	}
}