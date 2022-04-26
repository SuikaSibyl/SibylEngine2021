module;
#include <glm/glm.hpp>
#include <algorithm>
export module Math.Geometry;

namespace SIByL::Math
{
	export enum struct Relation
	{
		INTERSECT,
		PARALLEL,
		COPLANAR,
	};

	export struct IntersectResult
	{
		Relation relation;
		glm::vec3 position;
	};

	export struct Line
	{
		Line(glm::vec3 point_0, glm::vec3 point_1);
		auto parameterize(float t) const noexcept -> glm::vec3;

		// point-direction presentation
		glm::vec3 originPosition;
		glm::vec3 direction;

	};

	export struct Plane
	{

		// point-normal presentation
		glm::vec3 originPosition;
		glm::vec3 normal;
	};

	export inline auto intersect(Plane const& plane, Line const& line) noexcept -> IntersectResult;

	Line::Line(glm::vec3 point_0, glm::vec3 point_1)
	{
		originPosition = point_0;
		direction = glm::normalize(point_1 - point_0);
	}

	export inline auto clampLineUniformly(glm::vec3& a, glm::vec3& b) noexcept -> void
	{
		Line line(a, b);
		float bpara = (b - a).x / line.direction.x;

		float min_x = (0 - a.x) / line.direction.x;
		float max_x = (1 - a.x) / line.direction.x;
		if (min_x > max_x) std::swap(min_x, max_x);

		float min_y = (0 - a.y) / line.direction.y;
		float max_y = (1 - a.y) / line.direction.y;
		if (min_y > max_y) std::swap(min_y, max_y);

		float min_z = (0 - a.z) / line.direction.z;
		float max_z = (1 - a.z) / line.direction.z;
		if (min_z > max_z) std::swap(min_z, max_z);

		float near_t = std::max(min_x, std::max(min_y, min_z));
		float far_t = std::min(max_x, std::min(max_y, max_z));

		if (near_t > far_t)
		{
			a = { 0,0,0 };
			b = { 0,0,0 };
			return;
		}

		if (near_t > 0 && near_t < bpara) a = line.parameterize(near_t);
		if (far_t > 0 && far_t < bpara) b = line.parameterize(far_t);
		if (near_t > bpara || far_t < 0)
		{
			a = { 0,0,0 };
			b = { 0,0,0 };
		}
	}

	// ==============================================
	auto Line::parameterize(float t) const noexcept -> glm::vec3
	{
		return originPosition + direction * t;
	}

	inline auto intersect(Plane const& plane, Line const& line) noexcept -> IntersectResult
	{
		float ndd = glm::dot(plane.normal, line.direction);
		if (ndd == 0)
		{
			if (glm::dot(plane.normal, line.originPosition - plane.originPosition) != 0)
				return { Relation::PARALLEL };
			else
				return { Relation::COPLANAR };
		}
		else
		{
			float t = -1 * glm::dot(plane.normal, line.originPosition - plane.originPosition) / glm::dot(plane.normal, line.direction);
			return { Relation::INTERSECT, line.parameterize(t) };
		}
	}
}