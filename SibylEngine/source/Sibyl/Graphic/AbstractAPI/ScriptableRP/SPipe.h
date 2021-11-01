#pragma once

namespace SIByL
{
	class FrameBuffer;
	class RenderTarget;
	class Camera;

	namespace SRenderPipeline
	{
		class SPipe :public std::enable_shared_from_this<SPipe>
		{
		public:
			enum class PipeType
			{
				DrawPass,
				PostProcess,
			};
			virtual PipeType GetType() = 0;
#define DefinePipeType(type) virtual PipeType GetType() override {return type;}

			virtual ~SPipe() = default;
			virtual void Build() = 0;
			virtual void Attach() = 0;
			virtual void Draw() = 0;
			virtual void DrawImGui() = 0;

			virtual Ref<SPipe> AsSPipe() { return shared_from_this(); }

			virtual RenderTarget* GetRenderTarget(const std::string& name) = 0;
			virtual void SetInput(const std::string& name, RenderTarget* target) = 0;

			virtual void SetName(const std::string& name) { Name = name; };
			virtual const std::string& GetName() { return Name; }

		protected:
			std::string Name;
			std::vector<Ref<FrameBuffer>> m_Output;
		};

		class SPipeDrawPass :public SPipe
		{
		public:
			DefinePipeType(PipeType::DrawPass)

			virtual ~SPipeDrawPass() = default;
			virtual void Build() = 0;
			virtual void Attach() = 0;
			virtual void Draw() = 0;
			virtual void DrawImGui() = 0;

			virtual RenderTarget* GetRenderTarget(const std::string& name) = 0;
			virtual void SetInput(const std::string& name, RenderTarget* target) = 0;

			virtual Ref<SPipe> AsSPipe() { return std::dynamic_pointer_cast<SPipeDrawPass>(shared_from_this()); }

			Ref<Camera> mCamera;
		};

		class SPipePostProcess :public SPipe
		{
		public:
			DefinePipeType(PipeType::PostProcess)

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