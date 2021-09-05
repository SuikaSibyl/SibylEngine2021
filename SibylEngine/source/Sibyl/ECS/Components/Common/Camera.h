#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace SIByL
{
	class Camera;
	struct CameraComponent
	{
		CameraComponent(Ref<Camera> camera);

		Ref<Camera> m_Camrea;
		bool Primary = true;
		glm::mat4 m_Projection;
	};
}