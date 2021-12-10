#include "../../../Interface/Misc/IDevice.h"

namespace SIByL
{
	namespace Graphic
	{
		class GLDevice final : public IDevice
		{
		public:
			~GLDevice();

		public:
			virtual bool createProgram(const ShaderProgramDesc& desc, Ref<IShaderProgram> outProgram) noexcept override;

		};
	}
}