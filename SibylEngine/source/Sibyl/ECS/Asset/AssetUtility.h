#pragma once

#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Material.h"

namespace SIByL
{
	enum class AssetType
	{
		Unknown,
		Material,
	};

	std::string GetSuffix(const std::string& path);
	AssetType GetAssetType(const std::string& path);

	template<class T>
	Ref<T> GetAssetByPath(std::string path)
	{
		Ref<T> ref = Library<T>::Fetch(path);
		if (ref == nullptr)
		{
			Ref<Material> mat = CreateRef<Material>();
			MaterialSerializer serializer(mat);
			serializer.Deserialize(path);
			Library<T>::Push(path, mat);
			return mat;
		}
		else
		{
			return ref;
		}
	}
}