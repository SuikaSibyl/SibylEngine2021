module;
#include <cstdint>
export module RHI.IEnum;

namespace SIByL
{
	namespace RHI
	{
		export enum class API :uint32_t
		{
			VULKAN,
			DX12,
		};

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

		export enum class CommandPoolAttributeFlagBits : uint32_t
		{
			TRANSIENT = 0x00000001,
			RESET = 0x00000002,
		};
		export using CommandPoolAttributeFlags = uint32_t;

		export enum class PolygonMode : uint32_t
		{
			FILL,
			LINE,
			POINT,
		};

		export enum class CullMode : uint32_t
		{
			NONE,
			BACK,
			FRONT,
		};

		export enum class BlendOperator : uint32_t
		{
			ADD,
			SUBTRACT,
			REVERSE_SUBTRACT,
			MIN,
			MAX,
		};

		export enum class BlendFactor : uint32_t
		{
			ONE,
			ZERO,
			SRC_COLOR,
			ONE_MINUS_SRC_COLOR,
			DST_COLOR,
			ONE_MINUS_DST_COLOR,
			SRC_ALPHA,
			ONE_MINUS_SRC_ALPHA,
			DST_ALPHA,
			ONE_MINUS_DST_ALPHA,
			CONSTANT_COLOR,
			ONE_MINUS_CONSTANT_COLOR,
			CONSTANT_ALPHA,
			ONE_MINUS_CONSTANT_ALPHA,
			SRC_ALPHA_SATURATE,
			SRC1_COLOR,
			ONE_MINUS_SRC1_COLOR,
			SRC1_ALPHA,
			ONE_MINUS_SRC1_ALPHA,
		};

		export struct Extend
		{
			unsigned int width;
			unsigned int height;
		};

		export enum class PipelineState : uint32_t
		{
			VIEWPORT,
			LINE_WIDTH,
		};

		export enum class SampleCount : uint32_t
		{
			COUNT_1_BIT,
			COUNT_2_BIT,
			COUNT_4_BIT,
			COUNT_8_BIT,
			COUNT_16_BIT,
			COUNT_32_BIT,
			COUNT_64_BIT,
		};

		export enum class BufferUsageFlagBits : uint32_t
		{
			TRANSFER_SRC_BIT = 0x00000001,
			TRANSFER_DST_BIT = 0x00000002,
			UNIFORM_TEXEL_BUFFER_BIT = 0x00000004,
			STORAGE_TEXEL_BUFFER_BIT = 0x00000008,
			UNIFORM_BUFFER_BIT = 0x00000010,
			STORAGE_BUFFER_BIT = 0x00000020,
			INDEX_BUFFER_BIT = 0x00000040,
			VERTEX_BUFFER_BIT = 0x00000080,
			INDIRECT_BUFFER_BIT = 0x00000100,
			SHADER_DEVICE_ADDRESS_BIT = 0x00020000,
			FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
		};
		export using BufferUsageFlags = uint32_t;

		export enum class BufferShareMode :uint32_t
		{
			EXCLUSIVE = 0,
			CONCURRENT = 1,
		};

		export enum class MemoryPropertyFlagBits :uint32_t 
		{
			DEVICE_LOCAL_BIT = 0x00000001,
			HOST_VISIBLE_BIT = 0x00000002,
			HOST_COHERENT_BIT = 0x00000004,
			HOST_CACHED_BIT = 0x00000008,
			LAZILY_ALLOCATED_BIT = 0x00000010,
			PROTECTED_BIT = 0x00000020,
			DEVICE_COHERENT_BIT_AMD = 0x00000040,
			DEVICE_UNCACHED_BIT_AMD = 0x00000080,
			RDMA_CAPABLE_BIT_NV = 0x00000100,
			FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
		};
		export using MemoryPropertyFlags = uint32_t;

		export enum class CommandBufferUsageFlagBits :uint32_t
		{
			ONE_TIME_SUBMIT_BIT,
			RENDER_PASS_CONTINUE_BIT,
			SIMULTANEOUS_USE_BIT,
		};
		export using CommandBufferUsageFlags = uint32_t;

	}
}