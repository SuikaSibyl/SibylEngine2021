module;
#include <cstdint>
#include <string>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <thread>
#include <filesystem>
export module Core.Profiler;

namespace SIByL
{
	inline namespace Core
	{
		export struct ProfileResult
		{
			std::string tag;
			uint64_t start, end;
			uint32_t threadID;
		};

		export struct InstrumentationSession
		{
			InstrumentationSession(std::string const& n) :name(n) {}
			std::string name;
		};

		export struct Instrumentor
		{
		private:
			InstrumentationSession* currentSession;
			std::ofstream outputStream;
			int profileCount;

		public:
			Instrumentor()
				:currentSession(nullptr), profileCount(0)
			{
			}

			void beginSession(std::string const& name, std::filesystem::path filepath = "results.json")
			{
				outputStream.open("./profile/" / filepath);
				writeHeader();
				currentSession = new InstrumentationSession(name);
			}

			void endSession()
			{
				writeFooter();
				outputStream.close();
				delete currentSession;
				currentSession = nullptr;
				profileCount = 0;
			}

			void writeProfile(const ProfileResult& result)
			{
				if (profileCount++ > 0)
					outputStream << ",";

				std::string name = result.tag;
				std::replace(name.begin(), name.end(), '"', '\'');

				outputStream << "{";
				outputStream << "\"cat\":\"function\",";
				outputStream << "\"dur\":" << (result.end - result.start) << ',';
				outputStream << "\"name\":\"" << name << "\",";
				outputStream << "\"ph\":\"X\",";
				outputStream << "\"pid\":0,";
				outputStream << "\"tid\":" << result.threadID << ",";
				outputStream << "\"ts\":" << result.start;
				outputStream << "}";

				outputStream.flush();
			}

			void writeHeader()
			{
				outputStream << "{\"otherData\": {},\"traceEvents\":[";
				outputStream.flush();
			}

			void writeFooter()
			{
				outputStream << "]}";
				outputStream.flush();
			}

			static Instrumentor& get()
			{
				static Instrumentor instance;
				return instance;
			}
		};

		export struct InstrumentationTimer
		{
		public:
			InstrumentationTimer(const char* name)
				: name(name), stopped(false)
			{
				startTimepoint = std::chrono::high_resolution_clock::now();
			}

			~InstrumentationTimer()
			{
				if (!stopped)
					stop();
			}

			void stop()
			{
				auto endTimepoint = std::chrono::high_resolution_clock::now();

				uint64_t start = std::chrono::time_point_cast<std::chrono::microseconds>(startTimepoint).time_since_epoch().count();
				uint64_t end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();

				uint32_t threadID = std::hash<std::thread::id>{}(std::this_thread::get_id());
				Instrumentor::get().writeProfile({ name, start, end, threadID });

				stopped = true;
			}
		private:
			const char* name;
			std::chrono::time_point<std::chrono::high_resolution_clock> startTimepoint;
			bool stopped;
		};

		export inline auto PROFILE_BEGIN_SESSION(std::string const& name, std::filesystem::path filepath) noexcept -> void
		{
			Instrumentor::get().beginSession(name, filepath);
		}

		export inline auto PROFILE_END_SESSION() noexcept -> void
		{
			Instrumentor::get().endSession();
		}
	}
}