#pragma once

namespace SIByL
{
	namespace SGraphic
	{
        struct ResourceViewInfo
        {
            ResourceViewInfo() = default;

            // Init as texture
            ResourceViewInfo(uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
                : mostDetailedMip(mostDetailedMip), mipCount(mipCount), firstArraySlice(firstArraySlice), arraySize(arraySize) {}

            // Init as buffer
            ResourceViewInfo(uint32_t firstElement, uint32_t elementCount)
                : firstElement(firstElement), elementCount(elementCount) {}

            static const uint32_t kMaxPossible = -1;

            // Textures
            uint32_t mostDetailedMip = 0;
            uint32_t mipCount = kMaxPossible;
            uint32_t firstArraySlice = 0;
            uint32_t arraySize = kMaxPossible;

            // Buffers
            uint32_t firstElement = 0;
            uint32_t elementCount = kMaxPossible;

            bool operator==(const ResourceViewInfo& other) const
            {
                return (firstArraySlice == other.firstArraySlice)
                    && (arraySize == other.arraySize)
                    && (mipCount == other.mipCount)
                    && (mostDetailedMip == other.mostDetailedMip)
                    && (firstElement == other.firstElement)
                    && (elementCount == other.elementCount);
            }
        };


//        /** Abstracts API resource views.
//        */
//        template<typename ApiHandleType>
//        class ResourceView
//        {
//        public:
//            using ApiHandle = ApiHandleType;
//            using Dimension = ReflectionResourceType::Dimensions;
//            static const uint32_t kMaxPossible = -1;
//            virtual ~ResourceView();
//
//            ResourceView(ResourceWeakPtr& pResource, ApiHandle handle, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
//                : mApiHandle(handle), mpResource(pResource), mViewInfo(mostDetailedMip, mipCount, firstArraySlice, arraySize) {}
//
//            ResourceView(ResourceWeakPtr& pResource, ApiHandle handle, uint32_t firstElement, uint32_t elementCount)
//                : mApiHandle(handle), mpResource(pResource), mViewInfo(firstElement, elementCount) {}
//
//            ResourceView(ResourceWeakPtr& pResource, ApiHandle handle)
//                : mApiHandle(handle), mpResource(pResource) {}
//
//            /** Get the raw API handle.
//            */
//            const ApiHandle& getApiHandle() const { return mApiHandle; }
//
//            /** Get information about the view.
//            */
//            const ResourceViewInfo& getViewInfo() const { return mViewInfo; }
//
//            /** Get the resource referenced by the view.
//            */
//            Resource* getResource() const { return mpResource.lock().get(); }
//
//#if _ENABLE_CUDA
//            /** Get the CUDA device address for this view.
//            */
//            void* getCUDADeviceAddress() const
//            {
//                return mpResource.lock()->getCUDADeviceAddress(mViewInfo);
//            }
//#endif
//
//        protected:
//            ApiHandle mApiHandle;
//            ResourceViewInfo mViewInfo;
//            ResourceWeakPtr mpResource;
//        };
	}
}