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

		static void RegisterEarlyZOpaque();
		static void RegisterEarlyZDither();

		static void RegisterACES();

		static void RegisterTAA();
		static void RegisterFXAA();
		static void RegisterSharpen();
		static void RegisterMedianBlur();
		static void RegisterVignette();
		static void RegisterSSAO();
		static void RegisterSSAOCombine();

		static void RegisterBloomExtract();
		static void RegisterBloomCombine();
		static void RegisterBlur();
	};
}