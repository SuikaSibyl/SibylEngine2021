#pragma once

// For Bind By Application
#include "SIByLpch.h"
#include "Sibyl/Core/Application.h"
#include "Sibyl/Core/Layer.h"
#include "Sibyl/Core/Log.h"
#include "Sibyl/Core/Input.h"
#include "Sibyl/Core/KeyCodes.h"
#include "Sibyl/Core/MouseButtonCodes.h"
#include "Sibyl/ImGui/ImGuiLayer.h"

// EnterPoint
#include "Sibyl/Core/Instrumental.h"

// ImGui
#include "imgui.h"

// Scene
#include "Sibyl/ECS/Scene/Scene.h"
#include "Sibyl/ECS/Components/Components.h"

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// ECS
#include "Sibyl/ECS/Core/Entity.h"


// Render Basic
#include "Sibyl/Graphic/Core/Renderer/Renderer.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Shader.h"
#include "Sibyl/Graphic/Core/Geometry/TriangleMesh.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Texture.h"
#include "Sibyl/Graphic/Core/Renderer/Renderer2D.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"

// Components
#include <Sibyl/Graphic/Core/Camera.h>
#include <Sibyl/ECS/Components/ViewCameraController.h>