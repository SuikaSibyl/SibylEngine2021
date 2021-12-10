#pragma once

#include <iostream>

namespace SIByL
{
	namespace Graphic
	{
        enum class PipelineType
        {
            Unknown,
            Graphics,
            Compute,
            RayTracing,
            CountOf,
        };

        enum class StageType
        {
            Unknown,
            // Graphic Stage
            Vertex,
            Hull,
            Domain,
            Geometry,
            Fragment,
            // Compute
            Compute,
            // Ray tracing
            RayGeneration,
            Intersection,
            AnyHit,
            ClosestHit,
            Miss,
            Callable,
            Amplification,
            Mesh,
            CountOf,
        };

        enum class DeviceType
        {
            Unknown,
            Default,
            DirectX11,
            DirectX12,
            OpenGl,
            Vulkan,
            CPU,
            CUDA,
            CountOf,
        };

        enum class ProjectionStyle
        {
            Unknown,
            OpenGl,
            DirectX,
            Vulkan,
            CountOf,
        };

        /// The style of the binding
        enum class BindingStyle
        {
            Unknown,
            DirectX,
            OpenGl,
            Vulkan,
            CPU,
            CUDA,
            CountOf,
        };

        enum class PrimitiveType
        {
            Point, Line, Triangle, Patch
        };

        enum class PrimitiveTopology
        {
            TriangleList,
        };

        enum class ResourceState
        {
            Undefined,
            General,
            PreInitialized,
            VertexBuffer,
            IndexBuffer,
            ConstantBuffer,
            StreamOutput,
            ShaderResource,
            UnorderedAccess,
            RenderTarget,
            DepthRead,
            DepthWrite,
            Present,
            IndirectArgument,
            CopySource,
            CopyDestination,
            ResolveSource,
            ResolveDestination,
            AccelerationStructure,
            _Count
        };
        
        enum class ResourceType
        {
            Unknown,            ///< Unknown
            Buffer,             ///< A buffer (like a constant/index/vertex buffer)
            Texture1D,          ///< A 1d texture
            Texture2D,          ///< A 2d texture
            Texture3D,          ///< A 3d texture
            TextureCube,        ///< A cubemap consists of 6 Texture2D like faces
            CountOf,
        };

        struct ScissorRect
        {
            int32_t minX;
            int32_t minY;
            int32_t maxX;
            int32_t maxY;
        };

        struct Viewport
        {
            float originX = 0.0f;
            float originY = 0.0f;
            float extentX = 0.0f;
            float extentY = 0.0f;
            float minZ = 0.0f;
            float maxZ = 1.0f;
        };

    }
}