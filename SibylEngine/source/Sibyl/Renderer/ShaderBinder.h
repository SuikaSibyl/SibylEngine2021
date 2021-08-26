#pragma once

#include "ShaderData.h"

namespace SIByL
{
	struct ShaderUniformItem
	{
		std::string Name;
		ShaderDataType Type;
	};

	class ShaderBinder
	{
	public:
		static ShaderBinder* Create();
		virtual void BindFloat3() = 0;
		virtual void Bind() = 0;



	private:

	};
}