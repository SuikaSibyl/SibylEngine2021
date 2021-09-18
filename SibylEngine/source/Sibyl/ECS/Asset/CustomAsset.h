#pragma once

namespace SIByL
{
	class CustomAsset
	{
	public:
		void SetAssetDirty();
		void SetAssetUnDirty();
		void SetSavePath(const std::string& path);
		std::string GetSavePath();
		virtual void SaveAsset() = 0;

	protected:
		bool IsAssetDirty = false;
		std::string SavePath = "NONE";
	};
}