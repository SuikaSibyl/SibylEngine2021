#include "../../../Interface/Misc/IDevice.h"

namespace SIByL
{
	namespace Graphic
	{
		class GLDevice final : public IDevice
		{
		public:
			virtual ~GLDevice() noexcept = default;

		public:
			virtual bool createProgram(const ShaderProgramDesc& desc, Scope<IShaderProgram>& outProgram) noexcept override;

		};
	}
}