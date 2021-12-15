#pragma once

#include "../Misc/Define.h"

namespace SIByL
{
	namespace Graphic
	{
		class ProgramLayout;
		class IBlob;

		class IShaderComponent
		{
		public:
			virtual ~IShaderComponent() noexcept = default;

			virtual ProgramLayout* getLayout(
				sINT64    targetIndex = 0,
				IBlob** outDiagnostics = nullptr) noexcept = 0;

		};


	}
}