#include "SIByLpch.h"
#include "CustomAsset.h"

namespace SIByL
{
	void CustomAsset::SetAssetDirty()
	{
		IsAssetDirty = true;
	}

	void CustomAsset::SetAssetUnDirty()
	{
		IsAssetDirty = false;
	}

	void CustomAsset::SetSavePath(const std::string& path)
	{
		SavePath = path;
	}

	std::string CustomAsset::GetSavePath()
	{
		return SavePath;
	}
}