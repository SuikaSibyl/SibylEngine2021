export module RHI.GraphicContext.GL;
import Core.SObject;
import Core.Window;
import RHI.GraphicContext;

namespace SIByL
{
	namespace RHI
	{
		export class IGraphicContextGL :public IGraphicContext
		{
		public:
			virtual ~IGraphicContextGL() = default;
			// IGraphicContext
			virtual auto attachWindow(IWindow* window) noexcept -> void = 0;
			// IGraphicContextGL

		private:

		};
	}
}