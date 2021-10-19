#include "SIByLpch.h"
#include "SCacheAsset.h"

#include "Sibyl/Basic/Utils/StringUtils.h"

namespace SIByL
{
	SCacheAsset::SCacheAsset(const std::string& assetpath)
	{
		SetSavePath(AssetPathToCachePath(assetpath));
	}

	std::string SCacheAsset::AssetPathToCachePath(const std::string& path)
	{
		std::string standardPath = path;
		standardPath = replace_all(standardPath, "/", "_");
		standardPath = replace_all(standardPath, "\\", "_");
		standardPath = replace_all(standardPath, ".", "_");
		standardPath = "../Assets/Cache/" + standardPath + ".SCacheAsset";
		return standardPath;
	}

	void SCacheAsset::SaveCache()
	{
		LoadDataToBuffers();
		Serialize();
	}

	bool SCacheAsset::LoadCache()
	{
		bool success = Deserialize();
		if (success)
			LoadDataFromBuffers();
		return Deserialize();
	}

}