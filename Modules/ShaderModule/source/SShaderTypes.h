#pragma once

#include <iostream>

namespace SIByL
{
	namespace Shader
	{
		using SShaderInt = int64_t;

        struct BufferReflection;
        struct TypeLayoutReflection;
        struct TypeReflection;
        struct VariableLayoutReflection;
        struct VariableReflection;

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

            enum ScalarType
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

            Kind getKind();

            // only useful if `getKind() == Kind::Struct`
            unsigned int getFieldCount();

            VariableReflection* getFieldByIndex(unsigned int index);

            bool isArray();

            TypeReflection* unwrapArray();

            // only useful if `getKind() == Kind::Array`
            size_t getElementCount();

            size_t getTotalArrayElementCount();

            TypeReflection* getElementType();

            unsigned getRowCount();

            unsigned getColumnCount();

            ScalarType getScalarType();

            TypeReflection* getResourceResultType();

            //SlangResourceShape getResourceShape();

            //SlangResourceAccess getResourceAccess();

            //char const* getName();

            //unsigned int getUserAttributeCount();
            //UserAttribute* getUserAttributeByIndex(unsigned int index);
            //UserAttribute* findUserAttributeByName(char const* name);
        };
	}
}