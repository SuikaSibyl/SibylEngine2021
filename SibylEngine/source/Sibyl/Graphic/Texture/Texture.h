#pragma once

namespace SIByL
{
	class Texture
	{
	public:
		virtual ~Texture() = default;
		virtual uint32_t GetWidth() const = 0;
		virtual uint32_t GetHeight() const = 0;

		virtual void Bind(uint32_t slot) const = 0;

	};

	class Texture2D :public Texture
	{
	public:
		enum class Type
		{
			R8G8B8,
			R8G8B8A8,
		};

	public:
		static Ref<Texture2D> Create(const std::string& path);

	protected:
		Type m_Type;
	};
}