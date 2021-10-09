#include "SIByLpch.h"
#include "ShaderModule.h"

#include "Sibyl/Graphic/Core/Texture/Image.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Texture.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Material.h"
#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"

#include "DefaultShaders/ShaderRegister.h"

namespace SIByL
{
	//////////////////////////////////////
	// Specify Class Static Properties  //
	//////////////////////////////////////
	Ref<Shader>		ShaderModule::DefaultShader		= nullptr;
	Ref<Material>	ShaderModule::DefaultMaterial	= nullptr;
	Ref<Texture2D>	ShaderModule::DefaultTexture	= nullptr;


	//////////////////////////////////////
	//		 Static Functions		    //
	//////////////////////////////////////
	void ShaderModule::Init()
	{
		ShaderRegister::RegisterAll();
		

		Ref<Image> whiteImage = CreateRef<Image>(16, 16, 4, glm::vec4{ 1,1,1,1 });
		DefaultTexture = Texture2D::Create(whiteImage, "White");
		DefaultShader = Library<Shader>::Fetch("FILE=Shaders\\SIByL\\Texture");
		DefaultMaterial = CreateRef<Material>(DefaultShader);
		DefaultMaterial->SetFloat4("Color", { 1,0,1,1 });
		DefaultMaterial->SetTexture2D("Main", DefaultTexture);
	}

	Ref<Material> ShaderModule::GetDefaultMaterial()
	{
		return DefaultMaterial;
	}
}