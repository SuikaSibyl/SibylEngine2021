#pragma once

namespace SIByL
{
	class Shader
	{
	public:
		static Shader* Create();
		static Shader* Create(std::string vFile, std::string pFile);
		virtual void Use() = 0;
	private:

	};
}