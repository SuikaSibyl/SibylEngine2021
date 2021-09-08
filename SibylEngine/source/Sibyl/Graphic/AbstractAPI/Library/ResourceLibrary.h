#pragma once

#include "SIByLpch.h"

namespace SIByL
{
	template<class T>
	class Library
	{
	public:
		static void Push(const std::string& id, Ref<T> instance);
		static Ref<T> Fetch(const std::string& id);
		static void Remove(const std::string& id);

		static std::unordered_map<std::string, Ref<T>> Mapper;
	};

	template<class T>
	void Library<T>::Push(const std::string& id, Ref<T> instance)
	{
		if (Mapper.find(id) == Mapper.end())
		{
			Mapper[id] = instance;
		}
	}

	template<class T>
	Ref<T> Library<T>::Fetch(const std::string& id)
	{
		if (Mapper.find(id) != Mapper.end())
		{
			return Mapper[id];
		}
		else
			return nullptr;
	}

	template<class T>
	void Library<T>::Remove(const std::string& id)
	{
		if (Mapper.find(id) != Mapper.end())
		{
			Mapper.erase(id);
		}
	}

	template<class T>
	std::unordered_map<std::string, Ref<T>> Library<T>::Mapper;

}