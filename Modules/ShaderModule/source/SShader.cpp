#include "SShader.h"

#include "SlangMachine.h"

namespace SIByL
{
	void SShader::TestModule()
	{
		ShaderModule::SlangMachine machine;
		machine.CreateSession();
	}
}