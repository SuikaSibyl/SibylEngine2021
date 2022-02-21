module;

export module RHI.ITextureView;
import Core.SObject;

namespace SIByL
{
	namespace RHI
	{
		// Texture View
		// ╭──────────────┬───────────────────────────────╮
		// │  Vulkan	  │   vk::ImageView				  │
		// │  DirectX 12  │					              │
		// │  OpenGL      │			                      │
		// ╰──────────────┴───────────────────────────────╯
		export struct TextureViewDesc
		{

		};

		export class ITextureView :public SObject
		{
		public:
			ITextureView() = default;
			ITextureView(ITextureView&&) = default;
			ITextureView(ITextureView const&) = delete;
			virtual ~ITextureView() = default;


		};
	}
}
