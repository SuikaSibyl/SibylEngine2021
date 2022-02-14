module;
#include <cstdint>
export module RHI.IEnum;

namespace SIByL
{
	namespace RHI
	{
		export enum class DataType : uint32_t
		{
			None,
			Float,
			Float2,
			Float3,
			Float4,
			Mat3,
			Mat4,
			Int,
			Int2,
			Int3,
			Int4,
			Bool,
		};

		export inline auto sizeofDataType(DataType type) noexcept -> uint32_t;

		export enum class ResourceType : uint32_t
		{
			Buffer,                 ///< Buffer. Can be bound to all shader-stages
			Texture1D,              ///< 1D texture. Can be bound as render-target, shader-resource and UAV
			Texture2D,              ///< 2D texture. Can be bound as render-target, shader-resource and UAV
			Texture3D,              ///< 3D texture. Can be bound as render-target, shader-resource and UAV
			TextureCube,            ///< Texture-cube. Can be bound as render-target, shader-resource and UAV
			Texture2DMultisample,   ///< 2D multi-sampled texture. Can be bound as render-target, shader-resource and UAV
		};

		export enum class ResourceState : uint32_t
		{
			Undefined,
			PreInitialized,
			Common,
			VertexBuffer,
			ConstantBuffer,
			IndexBuffer,
			RenderTarget,
			UnorderedAccess,
			DepthStencil,
			ShaderResource,
			StreamOut,
			IndirectArg,
			CopyDest,
			CopySource,
			ResolveDest,
			ResolveSource,
			Present,
			GenericRead,
			Predication,
			PixelShader,
			NonPixelShader,
		};

		export enum class ResourceFormat : uint32_t
		{
			FORMAT_B8G8R8A8_RGB,
			FORMAT_B8G8R8A8_SRGB,
			FORMAT_R8G8B8A8_SRGB,
		};

		export enum class TopologyKind : uint32_t
		{
			PointList,
			LineList,
			LineStrip,
			TriangleList,
			TriangleStrip,
		};

		export enum class ShaderStage : uint32_t
		{
			VERTEX,
			FRAGMENT,
			GEOMETRY,
			TESSELLATION,
			COMPUTE,
			MESH,
		};

		export enum class QueueType : uint32_t
		{
			GRAPHICS,
			PRESENTATION,
		};
	}
}