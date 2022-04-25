export module GFX.RDG.Agency;

namespace SIByL::GFX::RDG
{
	export struct Agency
	{
		virtual auto onRegister(void* workshop) noexcept -> void = 0;
		virtual auto startWorkshopBuild(void* workshop) noexcept -> void = 0;
		virtual auto beforeCompile() noexcept -> void = 0;
		virtual auto beforeDivirtualizePasses() noexcept -> void = 0;
		virtual auto onInvalid() noexcept -> void = 0;
	};
}