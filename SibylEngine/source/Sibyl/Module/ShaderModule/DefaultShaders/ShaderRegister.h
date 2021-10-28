#pragma once

namespace SIByL
{
	class ShaderRegister
	{
	public:
		static void RegisterAll();

		static void RegisterUnlitTexture();

		static void RegisterLit();
		static void RegisterLitHair();

		static void RegisterACES();

		static void RegisterTAA();
	};
}