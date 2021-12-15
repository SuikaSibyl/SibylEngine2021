#pragma once

#include <iostream>

namespace SIByL
{
	namespace Graphic
	{
        typedef int                 sINT32;
        typedef unsigned int        sUINT32;
        typedef long long           sINT64;
        typedef unsigned long long  sUINT64;

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

        struct TypeReflection
        {
            enum class Kind
            {
                None,
                Struct,
                Array,
                Matrix,
                Vector,
                Scalar,
                ConstantBuffer,
                Resource,
                SamplerState,
                TextureBuffer,
                ShaderStorageBuffer,
                ParameterBlock,
                GenericTypeParameter,
                Interface,
                OutputStream,
                Specialized,
                Feedback,
            };

            enum ScalarType : unsigned int
            {
                None,
                Void,
                Bool,
                Int32,
                UInt32,
                Int64,
                UInt64,
                Float16,
                Float32,
                Float64,
                Int8,
                UInt8,
                Int16,
                UInt16,
            };

            //Kind getKind()
            //{
            //    return (Kind)spReflectionType_GetKind((SlangReflectionType*)this);
            //}

            //// only useful if `getKind() == Kind::Struct`
            //unsigned int getFieldCount()
            //{
            //    return spReflectionType_GetFieldCount((SlangReflectionType*)this);
            //}

            //VariableReflection* getFieldByIndex(unsigned int index)
            //{
            //    return (VariableReflection*)spReflectionType_GetFieldByIndex((SlangReflectionType*)this, index);
            //}

            //bool isArray() { return getKind() == TypeReflection::Kind::Array; }

            //TypeReflection* unwrapArray()
            //{
            //    TypeReflection* type = this;
            //    while (type->isArray())
            //    {
            //        type = type->getElementType();
            //    }
            //    return type;
            //}

            //// only useful if `getKind() == Kind::Array`
            //size_t getElementCount()
            //{
            //    return spReflectionType_GetElementCount((SlangReflectionType*)this);
            //}

            //size_t getTotalArrayElementCount()
            //{
            //    if (!isArray()) return 0;
            //    size_t result = 1;
            //    TypeReflection* type = this;
            //    for (;;)
            //    {
            //        if (!type->isArray())
            //            return result;

            //        result *= type->getElementCount();
            //        type = type->getElementType();
            //    }
            //}

            //TypeReflection* getElementType()
            //{
            //    return (TypeReflection*)spReflectionType_GetElementType((SlangReflectionType*)this);
            //}

            //unsigned getRowCount()
            //{
            //    return spReflectionType_GetRowCount((SlangReflectionType*)this);
            //}

            //unsigned getColumnCount()
            //{
            //    return spReflectionType_GetColumnCount((SlangReflectionType*)this);
            //}

            //ScalarType getScalarType()
            //{
            //    return (ScalarType)spReflectionType_GetScalarType((SlangReflectionType*)this);
            //}

            //TypeReflection* getResourceResultType()
            //{
            //    return (TypeReflection*)spReflectionType_GetResourceResultType((SlangReflectionType*)this);
            //}

            //SlangResourceShape getResourceShape()
            //{
            //    return spReflectionType_GetResourceShape((SlangReflectionType*)this);
            //}

            //SlangResourceAccess getResourceAccess()
            //{
            //    return spReflectionType_GetResourceAccess((SlangReflectionType*)this);
            //}

            //char const* getName()
            //{
            //    return spReflectionType_GetName((SlangReflectionType*)this);
            //}

            //unsigned int getUserAttributeCount()
            //{
            //    return spReflectionType_GetUserAttributeCount((SlangReflectionType*)this);
            //}
            //UserAttribute* getUserAttributeByIndex(unsigned int index)
            //{
            //    return (UserAttribute*)spReflectionType_GetUserAttribute((SlangReflectionType*)this, index);
            //}
            //UserAttribute* findUserAttributeByName(char const* name)
            //{
            //    return (UserAttribute*)spReflectionType_FindUserAttributeByName((SlangReflectionType*)this, name);
            //}
        };
    }
}