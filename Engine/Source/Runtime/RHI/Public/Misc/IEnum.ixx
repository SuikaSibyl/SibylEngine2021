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
			FORMAT_D32_SFLOAT, 
			FORMAT_D32_SFLOAT_S8_UINT, 
			FORMAT_D24_UNORM_S8_UINT
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

		export enum class DescriptorType :uint32_t
		{
			SAMPLER,
			COMBINED_IMAGE_SAMPLER,
			SAMPLED_IMAGE,
			STORAGE_IMAGE,
			UNIFORM_TEXEL_BUFFER,
			STORAGE_TEXEL_BUFFER,
			UNIFORM_BUFFER,
			STORAGE_BUFFER,
			UNIFORM_BUFFER_DYNAMIC,
			STORAGE_BUFFER_DYNAMIC,
			INPUT_ATTACHMENT,
			INLINE_UNIFORM_BLOCK_EXT,
			ACCELERATION_STRUCTURE_KHR,
			ACCELERATION_STRUCTURE_NV,
			MUTABLE_VALVE,
		};

		export enum class ShaderStageFlagBits :uint32_t
		{
			VERTEX_BIT = 0x00000001,
			TESSELLATION_CONTROL_BIT = 0x00000002,
			TESSELLATION_EVALUATION_BIT = 0x00000004,
			GEOMETRY_BIT = 0x00000008,
			FRAGMENT_BIT = 0x00000010,
			COMPUTE_BIT = 0x00000020,
			RAYGEN_BIT = 0x00000040,
			ANY_HIT_BIT = 0x00000080,
			CLOSEST_HIT_BIT = 0x00000100,
			MISS_BIT = 0x00000200,
			INTERSECTION_BIT = 0x00000400,
			CALLABLE_BIT = 0x00000800,
			TASK_BIT = 0x00001000,
			MESH_BIT = 0x00002080,
			SUBPASS_SHADING_BIT = 0x00004000,
		};
		export using ShaderStageFlags = uint32_t;

		export enum class PipelineBintPoint :uint32_t
		{
			GRAPHICS,
			COMPUTE,
			RAY_TRACING,
			SUBPASS_SHADING,
		};

		export enum class ImageTiling :uint32_t
		{
			OPTIMAL,
			LINEAR,
			DRM_FORMAT_MODIFIER,
		};

		export enum class ImageUsageFlagBits :uint32_t
		{
			TRANSFER_SRC_BIT = 0x00000001,
			TRANSFER_DST_BIT = 0x00000002,
			SAMPLED_BIT = 0x00000004,
			STORAGE_BIT = 0x00000008,
			COLOR_ATTACHMENT_BIT = 0x00000010,
			DEPTH_STENCIL_ATTACHMENT_BIT = 0x00000020,
			TRANSIENT_ATTACHMENT_BIT = 0x00000040,
			INPUT_ATTACHMENT_BIT = 0x00000080,
			FRAGMENT_DENSITY_MAP_BIT = 0x00000100,
			FRAGMENT_SHADING_RATE_ATTACHMENT_BIT = 0x00000200,
			INVOCATION_MASK_BIT = 0x00000400,
		};
		export using ImageUsageFlags = uint32_t;

		export enum class ImageLayout :uint32_t
		{
			UNDEFINED,
			GENERAL,
			COLOR_ATTACHMENT_OPTIMAL,
			DEPTH_STENCIL_ATTACHMENT_OPTIMA,
			DEPTH_STENCIL_READ_ONLY_OPTIMAL,
			SHADER_READ_ONLY_OPTIMAL,
			TRANSFER_SRC_OPTIMAL,
			TRANSFER_DST_OPTIMAL,
			PREINITIALIZED,
			DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
			DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
			DEPTH_ATTACHMENT_OPTIMAL,
			DEPTH_READ_ONLY_OPTIMAL,
			STENCIL_ATTACHMENT_OPTIMAL,
			STENCIL_READ_ONLY_OPTIMAL,
			PRESENT_SRC,
			SHARED_PRESENT,
			FRAGMENT_DENSITY_MAP_OPTIMAL,
			FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL,
			READ_ONLY_OPTIMAL,
			ATTACHMENT_OPTIMAL,
		};

		export inline auto flagBitSwitch(uint32_t const& input, uint32_t const& flag, uint32_t const& vendor_flag, uint32_t& target) noexcept -> void;
	}
}