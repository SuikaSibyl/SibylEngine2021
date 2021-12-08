#include "SlangMachine.h"

#include <iostream>

namespace SIByL
{
	namespace ShaderModule
	{
		ComPtr<slang::IGlobalSession> SlangMachine::sGlobalSession = nullptr;

		void SlangMachine::CreateSession()noexcept
		{
			slang::SessionDesc sessionDesc;

			slang::TargetDesc targetDesc;
			targetDesc.format = SLANG_GLSL;
			targetDesc.profile = GetGlobalSession()->findProfile("glsl_450");
			sessionDesc.targets = &targetDesc;
			sessionDesc.targetCount = 1;

			const char* searchPaths[] = { "G:/slang/examples/hello-world/" };
			sessionDesc.searchPaths = searchPaths;
			sessionDesc.searchPathCount = 1;

			GetGlobalSession()->createSession(sessionDesc, mSession.writeRef());

			slang::IModule* slangModule = nullptr;
			ComPtr<slang::IBlob> diagnostics;
			slangModule = mSession->loadModule("hello-world", diagnostics.writeRef());
			if (diagnostics)
			{
				fprintf(stderr, "%s\n", (const char*)diagnostics->getBufferPointer());
			}

			ComPtr<slang::IEntryPoint> computeEntryPoint;
			slangModule->findEntryPointByName("computeMain", computeEntryPoint.writeRef());

			slang::IComponentType* components[] = { slangModule, computeEntryPoint };
			ComPtr<slang::IComponentType> program;
			ComPtr<ISlangBlob> diagnosticsBlob;
			mSession->createCompositeComponentType(components, 2, program.writeRef(), diagnosticsBlob.writeRef());
			if (diagnosticsBlob)
			{
				fprintf(stderr, "%s\n", (const char*)diagnosticsBlob->getBufferPointer());
			}

			auto programReflection = program->getLayout();
			for (SlangUInt i = 0; i < programReflection->getEntryPointCount(); i++)
			{
				auto entryPointInfo = programReflection->getEntryPointByIndex(i);
				auto stage = entryPointInfo->getStage();
				
				ComPtr<ISlangBlob> kernelCode;
				ComPtr<ISlangBlob> diagnostics;
				auto compileResult = program->getEntryPointCode(
					(SlangInt)i, 0, kernelCode.writeRef(), diagnostics.writeRef());

				printf("%s\n", (const char*)kernelCode->getBufferPointer());

			}
		}

		auto SlangMachine::GetGlobalSession() noexcept->ComPtr<slang::IGlobalSession>
		{
			if (sGlobalSession == nullptr)
				slang::createGlobalSession(sGlobalSession.writeRef());

			return sGlobalSession;
		}

		auto SlangMachine::ReleaseGlobalSession() noexcept->void
		{
			sGlobalSession = nullptr;
		}

	}
}