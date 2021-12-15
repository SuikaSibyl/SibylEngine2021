#include "GLDevice.h"

#include <vector>

#include "../../../Interface/Misc/Define.h"
#include "../../../Interface/Shader/IShaderProgram.h"
#include "../../../Interface/Shader/IShaderComponent.h"
#include "../../../Interface/Shader/ShaderReflection.h"

#include "../Misc/GLContext.h"

namespace SIByL
{
	namespace Graphic
	{
		bool GLDevice::createProgram(const ShaderProgramDesc& desc, Scope<IShaderProgram>& outProgram) noexcept
		{
			unsigned int programID = glCreateProgram();
			ProgramLayout* programLayout = desc.program->getLayout();
			std::vector<GLuint> shaderIDs;
			for (unsigned int i = 0; i < programLayout->getEntryPointCount(); i++)
			{
				/// ...
				auto shaderID = 0;//
				shaderIDs.push_back(shaderID);
				glAttachShader(programID, shaderID);
			}
			glLinkProgram(programID);
			for (auto shaderID : shaderIDs)
				glDeleteShader(shaderID);

			// Check program link condition
			GLint success = GL_FALSE;
			glGetProgramiv(programID, GL_LINK_STATUS, &success);
			if (!success)
			{
				int maxSize = 0;
				glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &maxSize);

				auto infoBuffer = (char*)::malloc(maxSize);

				int infoSize = 0;
				glGetProgramInfoLog(programID, maxSize, &infoSize, infoBuffer);
				if (infoSize > 0)
				{
					fprintf(stderr, "%s", infoBuffer);
					OutputDebugStringA(infoBuffer);
				}

				::free(infoBuffer);

				glDeleteProgram(programID);
				return false;
			}

			//RefPtr<ShaderProgramImpl> program = new ShaderProgramImpl(m_weakRenderer, programID);
			outProgram.release();
			//outProgram = CreateScope<>
			return true;
		}

	}
}