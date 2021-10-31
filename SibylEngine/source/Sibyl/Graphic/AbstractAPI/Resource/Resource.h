#pragma once

namespace SIByL
{
	namespace SGraphic
	{
        class Texture;
        class Buffer;

        class Resource : public std::enable_shared_from_this<Resource>
        {
        public:
            enum class Type
            {
                Buffer,                 ///< Buffer. Can be bound to all shader-stages
                Texture1D,              ///< 1D texture. Can be bound as render-target, shader-resource and UAV
                Texture2D,              ///< 2D texture. Can be bound as render-target, shader-resource and UAV
                Texture3D,              ///< 3D texture. Can be bound as render-target, shader-resource and UAV
                TextureCube,            ///< Texture-cube. Can be bound as render-target, shader-resource and UAV
                Texture2DMultisample,   ///< 2D multi-sampled texture. Can be bound as render-target, shader-resource and UAV
            };

            /** Resource state. Keeps track of how the resource was last used*/
            enum class State : uint32_t
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

            using SharedPtr = std::shared_ptr<Resource>;
            using SharedConstPtr = std::shared_ptr<const Resource>;

            /** Default value used in create*() methods
            */
            //static const uint32_t kMaxPossible = RenderTargetView::kMaxPossible;

            Resource() = default;
            virtual ~Resource() = 0;

            /** Conversions to derived classes
            */
            std::shared_ptr<Texture> asTexture();
            std::shared_ptr<Buffer> asBuffer();

        };

        const std::string to_string(Resource::Type);
        const std::string to_string(Resource::State);
	}
}