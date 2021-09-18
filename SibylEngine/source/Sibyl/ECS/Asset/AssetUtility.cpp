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

	std::string PathToIdentifier(std::string path)
	{
		replace(path.begin(), path.end(), '/', '\\');
		std::string id = "FILE=" + path;
		return id;
	}

	std::string IdentifierToPath(std::string id)
	{
		SIByL_CORE_ASSERT(id.substr(0, 4) == "FILE", "Identifier of None Path");
		std::string path = id.substr(5, -1);
		return path;
	}

	bool IsIdentifierFromPath(std::string id)
	{
		return id.substr(0, 4) == "FILE";
	}
}