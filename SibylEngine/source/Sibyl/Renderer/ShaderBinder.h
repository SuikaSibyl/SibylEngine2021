#pragma once

namespace SIByL
{
	class ShaderUniformData
	{

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