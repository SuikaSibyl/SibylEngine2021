#pragma once

#include "SAsset.h"

namespace SIByL
{
	class SCacheAsset :public SAsset
	{
	public:
		SCacheAsset(const std::string& assetpath);
		virtual ~SCacheAsset() = default;

		virtual void LoadDataToBuffers() = 0;
		virtual void LoadDataFromBuffers() = 0;
		void SaveCache();
		bool LoadCache();

		static std::string AssetPathToCachePath(const std::string& path);
	};
}