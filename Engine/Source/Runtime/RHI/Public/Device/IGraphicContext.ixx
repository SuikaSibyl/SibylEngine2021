export module RHI.GraphicContext;
import Core.SObject;
import Core.Window;

namespace SIByL
{
	namespace RHI
	{
		export class IGraphicContext :public SObject
		{
		public:
			virtual ~IGraphicContext() = default;

			virtual auto attachWindow(IWindow* window) noexcept -> void = 0;

		private:

		};
	}
}