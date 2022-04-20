module;
#include <string>
#include <vector>
#include <algorithm>
export module RHI.IShaderReflection;
import Core.Log;
import RHI.IEnum;
import RHI.IDescriptorSetLayout;

namespace SIByL::RHI
{
	export struct DescriptorItem
	{
		unsigned set;
		unsigned binding;
		RHI::DescriptorType type;
		std::string name;
		ShaderStageFlags stageFlags = 0;

		bool operator <(DescriptorItem const& b) const { return (set < b.set) || ((set == b.set) && (binding < b.binding)); }
		bool operator ==(DescriptorItem const& b) const { return (set == b.set) && (binding == b.binding); }
		bool operator >(DescriptorItem const& b) const { return (set > b.set) || ((set == b.set) && (binding > b.binding)); }

	};

	export struct PushConstantItem
	{
		size_t index;
		size_t offset;
		size_t range;
		ShaderStageFlags stageFlags = 0;
	};

	export struct IShaderReflection
	{
	public:
		IShaderReflection() = default;
		virtual ~IShaderReflection() = default;
		IShaderReflection operator*(IShaderReflection const& t) const;
		auto findDescriptorItem() noexcept -> DescriptorItem*;
		auto toDescriptorSetLayoutDesc() noexcept -> DescriptorSetLayoutDesc;
		auto getPushConstantSize() noexcept -> size_t;

		std::vector<DescriptorItem> descriptorItems;
		std::vector<PushConstantItem> pushConstantItems;

	protected:
		auto sortDescriptorItems() noexcept -> void;
		auto sortPushConstantItems() noexcept -> void;
	};

	IShaderReflection IShaderReflection::operator*(IShaderReflection const& t) const
	{
		IShaderReflection result;
		// merge descriptorItems
		unsigned left_idx = 0;
		unsigned right_idx = 0;
		unsigned res_idx = 0;
		while (left_idx < descriptorItems.size() || right_idx < t.descriptorItems.size())
		{
			if (left_idx == descriptorItems.size())
				result.descriptorItems.emplace_back(t.descriptorItems[right_idx++]);
			else if (right_idx == t.descriptorItems.size())
				result.descriptorItems.emplace_back(descriptorItems[left_idx++]);
			else if (descriptorItems[left_idx] < t.descriptorItems[right_idx])
			{
				result.descriptorItems.emplace_back(descriptorItems[left_idx++]);
			}
			else if (descriptorItems[left_idx] > t.descriptorItems[right_idx])
			{
				result.descriptorItems.emplace_back(t.descriptorItems[right_idx++]);
			}
			else // ==
			{
				if (descriptorItems[left_idx].type == t.descriptorItems[right_idx].type)
				{
					result.descriptorItems.emplace_back(descriptorItems[left_idx++]);
					result.descriptorItems.back().stageFlags |= t.descriptorItems[right_idx++].stageFlags;
				}
				else
				{
					SE_CORE_ERROR("RHI :: Shader Reflection operator* merge failed, different types in merged shaders");
				}
			}
		}
		// merge pushConstantItems
		if (pushConstantItems.size() != 0 && t.pushConstantItems.size() == 0) result.pushConstantItems = pushConstantItems;
		else if (pushConstantItems.size() == 0 && t.pushConstantItems.size() != 0) result.pushConstantItems = t.pushConstantItems;
		else if (pushConstantItems.size() != 0 && t.pushConstantItems.size() != 0)
		{
			result.pushConstantItems = pushConstantItems;
			for (auto& item : result.pushConstantItems)
			{
				item.stageFlags |= t.pushConstantItems[0].stageFlags;
			}
		}

		return result;
	}

	auto IShaderReflection::getPushConstantSize() noexcept -> size_t
	{
		if (pushConstantItems.size() == 0) return 0;
		else return pushConstantItems.back().offset + pushConstantItems.back().range;
	}

	auto IShaderReflection::toDescriptorSetLayoutDesc() noexcept -> DescriptorSetLayoutDesc
	{
		RHI::DescriptorSetLayoutDesc descriptor_set_layout_desc;
		for (int i = 0; i < descriptorItems.size(); i++)
		{
			descriptor_set_layout_desc.perBindingDesc.emplace_back(
				descriptorItems[i].binding, 1, descriptorItems[i].type, descriptorItems[i].stageFlags, nullptr
			);
		}
		return descriptor_set_layout_desc;
	}

	bool compareDescriptorItem(DescriptorItem const& a, DescriptorItem const& b)
	{
		return (a.set < b.set) || ((a.set == b.set) && (a.binding < b.binding));
	}
	bool comparePushConstantItem(PushConstantItem const& a, PushConstantItem const& b)
	{
		return (a.index < b.index);
	}

	auto IShaderReflection::sortDescriptorItems() noexcept -> void
	{
		std::sort(descriptorItems.begin(), descriptorItems.end(), compareDescriptorItem);
	}

	auto IShaderReflection::sortPushConstantItems() noexcept -> void
	{
		std::sort(pushConstantItems.begin(), pushConstantItems.end(), comparePushConstantItem);
	}
}