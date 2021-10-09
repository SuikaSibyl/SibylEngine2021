#include "SIByLpch.h"
#include "ModuleManager.h"

#include <NetworkModule/include/NetworkModule.h>

namespace SIByL
{
	void ModuleManager::Init()
	{
		SIByLNetwork::NetworkModule::Init();
	}
}