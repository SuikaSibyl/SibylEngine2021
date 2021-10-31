//#pragma once
//
//namespace SIByL
//{
//    /** Reflection and layout information for a type in shader code.
//    */
//    class ReflectionType : public std::enable_shared_from_this<ReflectionType>
//    {
//    public:
//        using SharedPtr = std::shared_ptr<ReflectionType>;
//        using SharedConstPtr = std::shared_ptr<const ReflectionType>;
//
//        virtual ~ReflectionType() = default;
//
//        /** The kind of a type.
//        Every type has a kind, which specifies which subclass of `ReflectionType` it uses.
//        When adding new derived classes, this enumeration should be updated.
//        */
//        enum class Kind
//        {
//            Array,      ///< ReflectionArrayType
//            Struct,     ///< ReflectionStructType
//            Basic,      ///< ReflectionBasicType
//            Resource,   ///< ReflectionResourceType
//            Interface,  ///< ReflectionInterfaceType
//        };
//
//        /** Get the kind of this type.
//
//        The kind tells us if we have an array, structure, etc.
//        */
//        Kind getKind() const { return mKind; }
//
//        /** Dynamic-cast the current object to ReflectionResourceType
//        */
//        const ReflectionResourceType* asResourceType() const;
//
//        /** Dynamic-cast the current object to ReflectionBasicType
//        */
//        const ReflectionBasicType* asBasicType() const;
//
//        /** Dynamic-cast the current object to ReflectionStructType
//        */
//        const ReflectionStructType* asStructType() const;
//
//        /** Dynamic-cast the current object to ReflectionArrayType
//        */
//        const ReflectionArrayType* asArrayType() const;
//
//        /** Dynamic cast to ReflectionInterfaceType
//        */
//        const ReflectionInterfaceType* asInterfaceType() const;
//
//        /** "Unwrap" any array types to get to the non-array type underneath.
//
//        If `this` is not an array, then returns `this`.
//        If `this` is an array, then applies `unwrapArray` to its element type.
//        */
//        const ReflectionType* unwrapArray() const;
//
//        /** Get the total number of array elements represented by this type.
//
//        If `this` is not an array, then returns 1.
//        If `this` is an array, returns the number of elements times `getTotalArraySize()` for the element type.
//        */
//        uint32_t getTotalArrayElementCount() const;
//
//        /** Type to represent the byte size of a shader type.
//        */
//        typedef size_t ByteSize;
//
//        /** Get the size in bytes of instances of this type.
//
//        This function only counts uniform/ordinary data, and not resources like textures/buffers/samplers.
//        */
//        ByteSize getByteSize() const { return mByteSize; }
//
//        /** Find a field/member of this type with the given `name`.
//
//        If this type doesn't have fields/members, or doesn't have a field/member matching `name`, then returns null.
//        */
//        std::shared_ptr<const ReflectionVar> findMember(const std::string& name) const;
//
//        /** Get the (type and) offset of a field/member with the given `name`.
//
//        If this type doesn't have fields/members, or doesn't have a field/member matching `name`,
//        then logs an error and returns an invalid offset.
//        */
//        TypedShaderVarOffset getMemberOffset(const std::string& name) const;
//
//        /** Find a typed member/element offset corresponding to the given byte offset.
//        */
//        TypedShaderVarOffset findMemberByOffset(size_t byteOffset) const;
//
//        /** Get an offset that is zero bytes into this type.
//
//        Useful for turning a `ReflectionType` into a `TypedShaderVarOffset` so
//        that the `[]` operator can be used to look up members/elements.
//        */
//        TypedShaderVarOffset getZeroOffset() const;
//
//        /** Compare types for equality.
//
//        It is possible for two distinct `ReflectionType` instances to represent
//        the same type with the same layout. The `==` operator must be used to
//        tell if two types have the same structure.
//        */
//        virtual bool operator==(const ReflectionType& other) const = 0;
//
//        /** Compare types for inequality.
//        */
//        bool operator!=(const ReflectionType& other) const { return !(*this == other); }
//
//        /** A range of resources contained (directly or indirectly) in this type.
//
//        Different types will contain different numbers of resources, and those
//        resources will always be grouped into contiguous "ranges" that must be
//        allocated together in descriptor sets to allow them to be indexed.
//
//        Some examples:
//
//        * A basic type like `float2` has zero resoruce ranges.
//
//        * A resource type like `Texture2D` will have one resource range,
//          with a corresponding descriptor type and an array count of one.
//
//        * An array type like `float2[3]` or `Texture2D[4]` will have
//          the same number of ranges as its element type, but the count
//          of each range will be multiplied by the array element count.
//
//        * A structure type like `struct { Texture2D a; Texture2D b[3]; }`
//          will concatenate the resource ranges from its fields, in order.
//
//        The `ResourceRange` type is mostly an implementation detail
//        of `ReflectionType` that supports `ParameterBlock` and users
//        should probably not rely on this information.
//        */
//        struct ResourceRange
//        {
//            // TODO(tfoley) consider renaming this to `DescriptorRange`.
//
//            /** The type of descriptors that are stored in the range
//            */
//            DescriptorSet::Type descriptorType;
//
//            /** The total number of descriptors in the range.
//            */
//            uint32_t count;
//
//            /** If the enclosing type had its descriptors stored in
//            flattened arrays, where would this range start?
//
//            This is entirely an implementation detail of `ParameterBlock`.
//            */
//            uint32_t baseIndex;
//        };
//
//        /** Get the number of descriptor ranges contained in this type.
//        */
//        uint32_t getResourceRangeCount() const { return (uint32_t)mResourceRanges.size(); }
//
//        /** Get information on a contained descriptor range.
//        */
//        ResourceRange const& getResourceRange(uint32_t index) const { return mResourceRanges[index]; }
//
//        slang::TypeLayoutReflection* getSlangTypeLayout() const { return mpSlangTypeLayout; }
//
//    protected:
//        ReflectionType(Kind kind, ByteSize byteSize, slang::TypeLayoutReflection* pSlangTypeLayout)
//            : mKind(kind)
//            , mByteSize(byteSize)
//            , mpSlangTypeLayout(pSlangTypeLayout)
//        {}
//
//        Kind mKind;
//        ByteSize mByteSize = 0;
//        std::vector<ResourceRange> mResourceRanges;
//        slang::TypeLayoutReflection* mpSlangTypeLayout = nullptr;
//    };
//
//    /** Reflection object for resources
//    */
//    class ReflectionResourceType : public ReflectionType
//    {
//    public:
//        using SharedPtr = std::shared_ptr<ReflectionResourceType>;
//        using SharedConstPtr = std::shared_ptr<const ReflectionResourceType>;
//
//        /** Describes how the shader will access the resource
//        */
//        enum class ShaderAccess
//        {
//            Undefined,
//            Read,
//            ReadWrite
//        };
//
//        /** The expected return type
//        */
//        enum class ReturnType
//        {
//            Unknown,
//            Float,
//            Double,
//            Int,
//            Uint
//        };
//
//        /** The resource dimension
//        */
//        enum class Dimensions
//        {
//            Unknown,
//            Texture1D,
//            Texture2D,
//            Texture3D,
//            TextureCube,
//            Texture1DArray,
//            Texture2DArray,
//            Texture2DMS,
//            Texture2DMSArray,
//            TextureCubeArray,
//            AccelerationStructure,
//            Buffer,
//
//            Count
//        };
//
//        /** For structured-buffers, describes the type of the buffer
//        */
//        enum class StructuredType
//        {
//            Invalid,    ///< Not a structured buffer
//            Default,    ///< Regular structured buffer
//            Counter,    ///< RWStructuredBuffer with counter
//            Append,     ///< AppendStructuredBuffer
//            Consume     ///< ConsumeStructuredBuffer
//        };
//
//        /** The type of the resource
//        */
//        enum class Type
//        {
//            Texture,
//            StructuredBuffer,
//            RawBuffer,
//            TypedBuffer,
//            Sampler,
//            ConstantBuffer,
//            AccelerationStructure,
//        };
//
//        ///** Create a new object
//        //*/
//        //static SharedPtr create(
//        //    Type type, Dimensions dims, StructuredType structuredType, ReturnType retType, ShaderAccess shaderAccess,
//        //    slang::TypeLayoutReflection* pSlangTypeLayout);
//
//        /** For structured- and constant-buffers, set a reflection-type describing the buffer's layout
//        */
//        void setStructType(const ReflectionType::SharedConstPtr& pType);
//
//        /** Get the struct-type
//        */
//        const ReflectionType::SharedConstPtr& getStructType() const { return mpStructType; }
//
//        //const std::shared_ptr<const ParameterBlockReflection>& getParameterBlockReflector() const { return mpParameterBlockReflector; }
//        //void setParameterBlockReflector(const std::shared_ptr<const ParameterBlockReflection>& pReflector)
//        //{
//        //    mpParameterBlockReflector = pReflector;
//        //}
//
//        /** Get the dimensions
//        */
//        Dimensions getDimensions() const { return mDimensions; }
//
//        /** Get the structured-buffer type
//        */
//        StructuredType getStructuredBufferType() const { return mStructuredType; }
//
//        /** Get the resource return type
//        */
//        ReturnType getReturnType() const { return mReturnType; }
//
//        /** Get the required shader access
//        */
//        ShaderAccess getShaderAccess() const { return mShaderAccess; }
//
//        /** Get the resource type
//        */
//        Type getType() const { return mType; }
//
//        /** For structured- and constant-buffers, return the underlying type size, otherwise returns 0
//        */
//        size_t getSize() const { return mpStructType ? mpStructType->getByteSize() : 0; }
//
//        bool operator==(const ReflectionResourceType& other) const;
//        bool operator==(const ReflectionType& other) const override;
//    private:
//        ReflectionResourceType(Type type, Dimensions dims, StructuredType structuredType, ReturnType retType, ShaderAccess shaderAccess,
//            slang::TypeLayoutReflection* pSlangTypeLayout);
//
//        Dimensions mDimensions;
//        StructuredType mStructuredType;
//        ReturnType mReturnType;
//        ShaderAccess mShaderAccess;
//        Type mType;
//        ReflectionType::SharedConstPtr mpStructType;   // For constant- and structured-buffers
//        std::shared_ptr<const ParameterBlockReflection> mpParameterBlockReflector; // For constant buffers and parameter blocks
//    };
//
//    ///** Reflection and layout information for a type in shader code.
//    //*/
//    //class ReflectionType : public std::enable_shared_from_this<ReflectionType>
//    //{
//    //public:
//    //    using SharedPtr = std::shared_ptr<ReflectionType>;
//    //    using SharedConstPtr = std::shared_ptr<const ReflectionType>;
//
//    //    virtual ~ReflectionType() = default;
//
//    //    /** The kind of a type.
//
//    //    Every type has a kind, which specifies which subclass of `ReflectionType` it uses.
//
//    //    When adding new derived classes, this enumeration should be updated.
//    //    */
//    //    enum class Kind
//    //    {
//    //        Array,      ///< ReflectionArrayType
//    //        Struct,     ///< ReflectionStructType
//    //        Basic,      ///< ReflectionBasicType
//    //        Resource,   ///< ReflectionResourceType
//    //        Interface,  ///< ReflectionInterfaceType
//    //    };
//
//    //    /** Get the kind of this type.
//
//    //    The kind tells us if we have an array, structure, etc.
//    //    */
//    //    Kind getKind() const { return mKind; }
//
//    //    /** Dynamic-cast the current object to ReflectionResourceType
//    //    */
//    //    const ReflectionResourceType* asResourceType() const;
//
//    //    /** Dynamic-cast the current object to ReflectionBasicType
//    //    */
//    //    const ReflectionBasicType* asBasicType() const;
//
//    //    /** Dynamic-cast the current object to ReflectionStructType
//    //    */
//    //    const ReflectionStructType* asStructType() const;
//
//    //    /** Dynamic-cast the current object to ReflectionArrayType
//    //    */
//    //    const ReflectionArrayType* asArrayType() const;
//
//    //    /** Dynamic cast to ReflectionInterfaceType
//    //    */
//    //    const ReflectionInterfaceType* asInterfaceType() const;
//
//    //    /** "Unwrap" any array types to get to the non-array type underneath.
//
//    //    If `this` is not an array, then returns `this`.
//    //    If `this` is an array, then applies `unwrapArray` to its element type.
//    //    */
//    //    const ReflectionType* unwrapArray() const;
//
//    //    /** Get the total number of array elements represented by this type.
//
//    //    If `this` is not an array, then returns 1.
//    //    If `this` is an array, returns the number of elements times `getTotalArraySize()` for the element type.
//    //    */
//    //    uint32_t getTotalArrayElementCount() const;
//
//    //    /** Type to represent the byte size of a shader type.
//    //    */
//    //    typedef size_t ByteSize;
//
//    //    /** Get the size in bytes of instances of this type.
//
//    //    This function only counts uniform/ordinary data, and not resources like textures/buffers/samplers.
//    //    */
//    //    ByteSize getByteSize() const { return mByteSize; }
//
//    //    /** Find a field/member of this type with the given `name`.
//
//    //    If this type doesn't have fields/members, or doesn't have a field/member matching `name`, then returns null.
//    //    */
//    //    std::shared_ptr<const ReflectionVar> findMember(const std::string& name) const;
//
//    //    /** Get the (type and) offset of a field/member with the given `name`.
//
//    //    If this type doesn't have fields/members, or doesn't have a field/member matching `name`,
//    //    then logs an error and returns an invalid offset.
//    //    */
//    //    TypedShaderVarOffset getMemberOffset(const std::string& name) const;
//
//    //    /** Find a typed member/element offset corresponding to the given byte offset.
//    //    */
//    //    TypedShaderVarOffset findMemberByOffset(size_t byteOffset) const;
//
//    //    /** Get an offset that is zero bytes into this type.
//
//    //    Useful for turning a `ReflectionType` into a `TypedShaderVarOffset` so
//    //    that the `[]` operator can be used to look up members/elements.
//    //    */
//    //    TypedShaderVarOffset getZeroOffset() const;
//
//    //    /** Compare types for equality.
//
//    //    It is possible for two distinct `ReflectionType` instances to represent
//    //    the same type with the same layout. The `==` operator must be used to
//    //    tell if two types have the same structure.
//    //    */
//    //    virtual bool operator==(const ReflectionType& other) const = 0;
//
//    //    /** Compare types for inequality.
//    //    */
//    //    bool operator!=(const ReflectionType& other) const { return !(*this == other); }
//
//    //    /** A range of resources contained (directly or indirectly) in this type.
//
//    //    Different types will contain different numbers of resources, and those
//    //    resources will always be grouped into contiguous "ranges" that must be
//    //    allocated together in descriptor sets to allow them to be indexed.
//
//    //    Some examples:
//
//    //    * A basic type like `float2` has zero resoruce ranges.
//
//    //    * A resource type like `Texture2D` will have one resource range,
//    //      with a corresponding descriptor type and an array count of one.
//
//    //    * An array type like `float2[3]` or `Texture2D[4]` will have
//    //      the same number of ranges as its element type, but the count
//    //      of each range will be multiplied by the array element count.
//
//    //    * A structure type like `struct { Texture2D a; Texture2D b[3]; }`
//    //      will concatenate the resource ranges from its fields, in order.
//
//    //    The `ResourceRange` type is mostly an implementation detail
//    //    of `ReflectionType` that supports `ParameterBlock` and users
//    //    should probably not rely on this information.
//    //    */
//    //    struct ResourceRange
//    //    {
//    //        // TODO(tfoley) consider renaming this to `DescriptorRange`.
//
//    //        /** The type of descriptors that are stored in the range
//    //        */
//    //        DescriptorSet::Type descriptorType;
//
//    //        /** The total number of descriptors in the range.
//    //        */
//    //        uint32_t count;
//
//    //        /** If the enclosing type had its descriptors stored in
//    //        flattened arrays, where would this range start?
//
//    //        This is entirely an implementation detail of `ParameterBlock`.
//    //        */
//    //        uint32_t baseIndex;
//    //    };
//
//    //    /** Get the number of descriptor ranges contained in this type.
//    //    */
//    //    uint32_t getResourceRangeCount() const { return (uint32_t)mResourceRanges.size(); }
//
//    //    /** Get information on a contained descriptor range.
//    //    */
//    //    ResourceRange const& getResourceRange(uint32_t index) const { return mResourceRanges[index]; }
//
//    //    slang::TypeLayoutReflection* getSlangTypeLayout() const { return mpSlangTypeLayout; }
//
//    //protected:
//    //    ReflectionType(Kind kind, ByteSize byteSize, slang::TypeLayoutReflection* pSlangTypeLayout)
//    //        : mKind(kind)
//    //        , mByteSize(byteSize)
//    //        , mpSlangTypeLayout(pSlangTypeLayout)
//    //    {}
//
//    //    Kind mKind;
//    //    ByteSize mByteSize = 0;
//    //    std::vector<ResourceRange> mResourceRanges;
//    //    slang::TypeLayoutReflection* mpSlangTypeLayout = nullptr;
//    //};
//
//}