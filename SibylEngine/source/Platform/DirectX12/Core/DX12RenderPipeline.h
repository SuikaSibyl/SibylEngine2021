#pragma once

namespace SIByL
{
	class DX12RenderPipeline
	{
	public:
		static DX12RenderPipeline* Main;
		DX12RenderPipeline();
		void static DrawFrame() { Main->DrawFrameImpl(); }

	protected:
		void static DrawFrameImpl();

	private:
	};
}