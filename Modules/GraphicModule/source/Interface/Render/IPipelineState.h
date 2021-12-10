#pragma once

namespace SIByL
{
	namespace Graphic
	{
		class IShaderProgram;
		class IFramebufferLayout;

		class IPipelineState
		{
		public:
			virtual ~IPipelineState() noexcept = default;



		};

		struct GraphicsPipelineStateDesc
		{
			IShaderProgram* program = nullptr;

			//IInputLayout* inputLayout = nullptr;
			IFramebufferLayout* framebufferLayout = nullptr;
			//PrimitiveType       primitiveType = PrimitiveType::Triangle;
			//DepthStencilDesc    depthStencil;
			//RasterizerDesc      rasterizer;
			//BlendDesc           blend;
		};

		struct ComputePipelineStateDesc
		{
			IShaderProgram* program;
		};


	}
}