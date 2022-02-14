module;
#include <slang.h>
#include <slang-com-ptr.h>
#include <string_view>
module RHI.ICompileSession;
import Core.SObject;
import Core.Log;
import Core.MemoryManager;
import RHI.IGlobalSession;
import RHI.SlangUtility;

using Slang::ComPtr;

namespace SIByL::RHI::SLANG
{
	char const* search_paths[] =
	{
		"../Engine/Shaders/Test",
	};

	ICompileSession::ICompileSession()
	{
		slang::SessionDesc sessionDesc = {};
		slang::TargetDesc targetDesc = {};
		targetDesc.format = SLANG_SPIRV;
		targetDesc.profile = IGlobalSession::instance()->getGlobalSession()->findProfile("glsl440");

		sessionDesc.targets = &targetDesc;
		sessionDesc.targetCount = 1;
		sessionDesc.searchPathCount = sizeof(search_paths) / sizeof(char const*);
		sessionDesc.searchPaths = search_paths;
		IGlobalSession::instance()->getGlobalSession()->createSession(sessionDesc, session.writeRef());
	}

	auto ICompileSession::loadModule(std::string_view module_name, std::string_view entry_point) noexcept -> bool
	{
		// load the module
		slang::IModule* slangModule = nullptr;
		{
			ComPtr<slang::IBlob> diagnosticBlob;
			slangModule = session->loadModule(module_name.data(), diagnosticBlob.writeRef());
			diagnoseIfNeeded(diagnosticBlob);
			if (!slangModule)
				return false;
		}

		// look up entry points
		ComPtr<slang::IEntryPoint> computeEntryPoint;
		slangModule->findEntryPointByName(entry_point.data(), computeEntryPoint.writeRef());

		// create program
		slang::IComponentType* components[] = { slangModule, computeEntryPoint };
		ComPtr<slang::IComponentType> program;
		session->createCompositeComponentType(components, 2, program.writeRef());

		// do compile
		int entryPointIndex = 0; // only one entry point
		int targetIndex = 0; // only one target
		ComPtr<slang::IBlob> kernelBlob;
		ComPtr<slang::IBlob> diagnostics;
		program->getEntryPointCode(
			entryPointIndex,
			targetIndex,
			kernelBlob.writeRef(),
			diagnostics.writeRef());
		if (diagnostics != nullptr)
		{
			SE_CORE_ERROR("SLANG :: {0}", (const char*)diagnostics->getBufferPointer());
		}

		// get layout
		slang::ProgramLayout* layout = program->getLayout();

		// get kernal
		char* kernalBytecode = (char*)Memory::instance()->allocate(kernelBlob->getBufferSize());
		memcpy(kernalBytecode, (const char*)kernelBlob->getBufferPointer(), kernelBlob->getBufferSize());

		return true;
	}
}