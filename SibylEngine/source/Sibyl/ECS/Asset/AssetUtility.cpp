#include "SIByLpch.h"
#include "AssetUtility.h"

namespace SIByL
{
	std::string GetSuffix(const std::string& path)
	{
		return path.substr(path.find_last_of(".") + 1);
	}

	AssetType GetAssetType(const std::string& path)
	{
		std::string suffix = GetSuffix(path);
		if (suffix == "mat")
		{
			return AssetType::Material;
		}
		else
		{
			return AssetType::Unknown;
		}
	}

}