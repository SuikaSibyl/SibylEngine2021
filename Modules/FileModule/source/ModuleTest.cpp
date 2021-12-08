#include "ModuleTest.h"

#include <Core/module.h>

#include "SFileSystem.h"
#include "SCache.h"

namespace SIByL
{
	namespace File
	{
		void ModuleTest::Test()
		{
			TimeStamp lastWrite = FileSystem::GetLastWriteTime("C:/Users/fhm/Desktop/RelatedWork/test.txt");
			TimeStamp lastWrite2 = FileSystem::GetLastWriteTime("C:/Users/fhm/Desktop/RelatedWork/test2.txt");
			S_CORE_DEBUG("Last write time: {0}", lastWrite);
			S_CORE_DEBUG("Last write time: {0}", lastWrite2);

			struct ShaderInfo
			{
				float Time;
				float Direct;
			};

			std::vector<ShaderInfo> info = { { 1,2 }, {3,4} };
			SCache cache("C:/Users/fhm/Desktop/RelatedWork/cache1.txt");
			cache.PushStructVector(info);
			cache.Serialize();

			std::vector<ShaderInfo> infoR;
			SCache cacheR("C:/Users/fhm/Desktop/RelatedWork/cache1.txt");
			cacheR.Deserialize();
			cacheR.PullStructVector(infoR);
		}

	}
}