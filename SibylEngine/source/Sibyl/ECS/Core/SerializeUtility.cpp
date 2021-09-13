#include "SIByLpch.h"
#include "SerializeUtility.h"

#include "Sibyl/Graphic/Core/Geometry/TriangleMesh.h"

namespace SIByL
{
	YAML::Emitter& operator<<(YAML::Emitter& out, Ref<TriangleMesh> mesh)
	{
		out << mesh->m_Path;
		return out;
	}
}