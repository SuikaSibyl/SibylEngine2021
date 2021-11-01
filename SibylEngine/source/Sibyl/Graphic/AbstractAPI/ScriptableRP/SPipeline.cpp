#include "SIByLpch.h"
#include "SPipeline.h"

#include "SPipe.h"

namespace SIByL
{
	namespace SRenderPipeline
	{
		void SPipeline::Build()
		{
			for (Ref<SPipe>& pipe : mPipes)
			{
				if (pipe->GetType() == SPipe::PipeType::DrawPass)
				{
					mDrawPassNames.push_back(pipe->GetName());
				}
			}

		}

		void SPipeline::Run()
		{
			for (Ref<SPipe>& pipe : mPipes)
			{
				pipe->Draw();
			}
		}

		void SPipeline::InsertPipe(Ref<SPipe> pipe, const std::string& name)
		{
			pipe->Attach();
			pipe->SetName(name);
			mPipes.push_back(pipe);
			mPipeMapper[name] = pipe;
		}

		void SPipeline::AttachPipes(const std::string& pipeOut, const std::string& outputName,
			const std::string& pipeIn, const std::string& inputName)
		{
			Ref<SPipe> PipeOut = GetPipe(pipeOut);
			Ref<SPipe> Pipein = GetPipe(pipeIn);
			Pipein->SetInput(inputName, PipeOut->GetRenderTarget(outputName));
		}

		Ref<SPipe> SPipeline::GetPipe(const std::string& name)
		{
			auto& iter = mPipeMapper.find(name);
			if (iter != mPipeMapper.end())
			{
				return iter->second;
			}
			else
			{
				SIByL_CORE_WARN("No Pipe named: " + name + ", nullptr is returned!");
				return nullptr;
			}
		}

	}
}