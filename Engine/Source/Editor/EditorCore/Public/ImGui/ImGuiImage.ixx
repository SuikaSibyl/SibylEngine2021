module;
#include <imgui.h>
export module Editor.ImImage;
import RHI.ITexture;

namespace SIByL::Editor
{
	export struct ImImage
	{
		virtual auto getImTextureID() noexcept -> ImTextureID = 0;
	};
}