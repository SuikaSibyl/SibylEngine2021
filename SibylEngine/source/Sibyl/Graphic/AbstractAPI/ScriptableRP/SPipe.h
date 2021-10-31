#pragma once

namespace SIByL
{
	class FrameBuffer;
	class RenderTarget;

	namespace SRenderPipeline
	{
		class SPipe
		{
		public:
			virtual ~SPipe() = default;
			virtual void Build() = 0;
			virtual void Attach() = 0;
			virtual void Draw() = 0;
			virtual void DrawImGui() = 0;

			
			virtual RenderTarget* GetRenderTarget(const std::string& name) = 0;
			virtual void SetInput(const std::string& name, RenderTarget* target) = 0;

		protected:
			std::vector<Ref<FrameBuffer>> m_Output;
		};

		class SPipeDraw :public SPipe
		{
		public:
			virtual ~SPipeDraw() = default;
			virtual void Build() = 0;
			virtual void Attach() = 0;
			virtual void Draw() = 0;
			virtual void DrawImGui() = 0;

			virtual RenderTarget* GetRenderTarget(const std::string& name) = 0;
			virtual void SetInput(const std::string& name, RenderTarget* target) = 0;
		};

		class SPipePostProcess :public SPipe
		{
		public:
			virtual ~SPipePostProcess() = default;
			virtual void Build() = 0;
			virtual void Attach() = 0;
			virtual void Draw() = 0;
			virtual void DrawImGui() = 0;

			virtual RenderTarget* GetRenderTarget(const std::string& name) = 0;
			virtual void SetInput(const std::string& name, RenderTarget* target) = 0;
		};

#define SPipeBegin(Type)    static Ref<SPipe> Create()\
							{\
								Ref<SPipe> pipe;\
								pipe.reset(new Type());\
								pipe->Build();\
								return pipe;\
							}\

	}
}