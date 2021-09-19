#include "SIByLpch.h"
#include "ResourceLibrary.h"

#include "Sibyl/ECS/Asset/AssetUtility.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Texture.h"

namespace SIByL
{
	template< >
	Ref<Texture2D> Library<Texture2D>::Fetch(const std::string& id)
	{
		if (Mapper.find(id) != Mapper.end())
		{
			return Mapper[id];
		}
		else
		{
			if (IsIdentifierFromPath(id))
			{
				Ref<Texture2D> texture = Texture2D::Create(IdentifierToPath(id));
				return texture;
			}
		}

		return nullptr;
	}
}