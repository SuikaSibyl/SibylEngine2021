#pragma once

namespace SIByL
{
	class Shader
	{
	public:
		static Shader* Create();
		virtual void Use() = 0;
	private:

	};
}