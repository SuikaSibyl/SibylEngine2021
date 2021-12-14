#pragma once

#include "../../../../Core/module.h"
#include "ShaderOffset.h"

namespace SIByL
{
	namespace Graphic
	{
		class IBufferResource;

		class IShaderObject
		{
		public:
			virtual ~IShaderObject() noexcept = default;

			Scope<IShaderObject> getObject(ShaderOffset const& offset);

			virtual auto getObject(ShaderOffset const& offset, Scope<IShaderObject>& object)noexcept->bool = 0;
			
			virtual size_t getSize()noexcept = 0;

			/// Use the provided constant buffer instead of the internally created one.
			virtual bool setConstantBufferOverride(IBufferResource* constantBuffer)noexcept = 0;
		};

		
	}
}