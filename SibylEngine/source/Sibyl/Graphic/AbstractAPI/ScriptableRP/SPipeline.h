#pragma once

namespace SIByL
{
	namespace SRenderPipeline
	{
		class SPipe;
		class SPipeline
		{
		public:
			void Build();
			void Run();

			void InsertPipe(Ref<SPipe> pipe, const std::string& name);
			void AttachPipes(const std::string& pipeOut, const std::string& outputName,
				const std::string& pipeIn, const std::string& inputName);

			Ref<SPipe> GetPipe(const std::string& name);
			std::vector<std::string>& GetDrawPassesNames() { return mDrawPassNames; }

		protected:
			std::vector<Ref<SPipe>> mPipes;
			std::unordered_map<std::string, Ref<SPipe>> mPipeMapper;
			std::vector<std::string> mDrawPassNames;
		};
	}
}