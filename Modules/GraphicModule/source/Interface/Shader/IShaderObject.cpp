#include "IShaderObject.h"

namespace SIByL
{
	namespace Graphic
	{
		Scope<IShaderObject> IShaderObject::getObject(ShaderOffset const& offset)
		{
			Scope<IShaderObject> object = nullptr;
			S_CORE_ASSERT(getObject(offset, object), "Shader Object create failed");
			return object;
		}

	}
}