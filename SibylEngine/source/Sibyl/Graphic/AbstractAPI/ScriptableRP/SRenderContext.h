#pragma once

namespace SIByL
{
	namespace SRenderPipeline
	{
		class SRenderContext
		{
		public:
			using pair_uint = std::pair<unsigned int, unsigned int>;
			static pair_uint GetScreenSize() { return pair_uint{ ScreenWidth, ScreenHeight }; }
			static void SetScreenSize(pair_uint size) { ScreenWidth = size.first; ScreenHeight = size.second; }

		private:
			static unsigned int ScreenWidth, ScreenHeight;
		};

	}
}