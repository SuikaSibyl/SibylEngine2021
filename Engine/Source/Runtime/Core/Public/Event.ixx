module;
#include <string>
#include <iostream>
#include <functional>
#include <sstream>
export module Core.Event;

#define BIT(x) (1 << x)

namespace SIByL
{
	inline namespace Core
	{
		export enum class EventType
		{
			None = 0,
			WindowClose, WindowResize, WindowFocus, WindowLostFocus, WindowMoved,
			AppTick, AppUpdate, AppRender,
			KeyPressed, KeyReleased, KeyTyped,
			MouseButtonPressed, MouseButtonReleased, MouseMoved, MouseScrolled,
		};

		export enum EventCategory
		{
			None = 0,
			EventCategoryApplication = BIT(0),
			EventCategoryInput = BIT(1),
			EventCategoryKeyboard = BIT(2),
			EventCategoryMouse = BIT(3),
			EventCategoryMouseButton = BIT(4),
		};

		export class Event
		{
			friend class EventDispatcher;

		public:
			virtual EventType getEventType() const = 0;
			virtual const char* getName() const = 0;
			virtual int getCategoryFlags() const = 0;
			virtual std::string toString() const { return getName(); }

			inline bool isInCategory(EventCategory category)
			{
				return getCategoryFlags() & (int)category;
			}

			bool handled = false;
		};

		export class EventDispatcher
		{
			template<typename T>
			using EventFn = std::function<bool(T&)>;

		public:
			EventDispatcher(Event& event)
				:event_to_handle(event)
			{
			}

			template<typename T>
			bool Dispatch(EventFn<T> func)
			{
				if (event_to_handle.getEventType() == T::getStaticType())
				{
					event_to_handle.handled = func(*(T*)&event_to_handle);
					return true;
				}
				return false;
			}

		private:
			Event& event_to_handle;
		};

		export inline std::ostream& operator<<(std::ostream& os, const Event& e)
		{
			return os << e.toString();
		}


#define EVENT_CLASS_TYPE(type) static  EventType getStaticType() {return EventType::##type;}\
								virtual EventType getEventType() const override {return getStaticType();}\
								virtual const char* getName() const override {return #type;}

#define EVENT_CLASS_CATEGORY(category) virtual int getCategoryFlags() const override {return category;}

		// Key events
		export class KeyEvent : public Event
		{
		public:
			inline int GetKeyCode() const { return m_KeyCode; }

			EVENT_CLASS_CATEGORY(EventCategory::EventCategoryKeyboard | EventCategory::EventCategoryInput);

		protected:
			KeyEvent(int keycode)
				:m_KeyCode(keycode) {}

			int m_KeyCode;
		};

		export class KeyPressedEvent :public KeyEvent
		{
		public:
			KeyPressedEvent(int keycode, int repeatCount)
				:KeyEvent(keycode), m_RepeatCount(repeatCount) {}

			inline int GetRepeatCount() const { return m_RepeatCount; }

			std::string toString() const override
			{
				std::stringstream ss;
				ss << "KeyPressedEvent: " << m_KeyCode << " (" << m_RepeatCount << " repeats)";
				return ss.str();
			}

			EVENT_CLASS_TYPE(KeyPressed)

		private:
			int m_RepeatCount;
		};

		export class KeyReleasedEvent :public KeyEvent
		{
		public:
			KeyReleasedEvent(int keycode)
				:KeyEvent(keycode) {}

			std::string toString() const override
			{
				std::stringstream ss;
				ss << "KeyReleasedEvent: " << m_KeyCode;
				return ss.str();
			}

			EVENT_CLASS_TYPE(KeyReleased);
		};

		export class KeyTypedEvent :public KeyEvent
		{
		public:
			KeyTypedEvent(unsigned int keycode)
				:KeyEvent(keycode) {}

			std::string toString() const override
			{
				std::stringstream ss;
				ss << "KeyTypedEvent: " << m_KeyCode;
				return ss.str();
			}

			EVENT_CLASS_TYPE(KeyTyped);
		};

		// Application Events

		export class WindowResizeEvent :public Event
		{
		public:
			WindowResizeEvent(unsigned int width, unsigned int height)
				:m_Width(width), m_Height(height)
			{}

			inline unsigned int GetWidth() const { return m_Width; }
			inline unsigned int GetHeight() const { return m_Height; }

			std::string toString() const override
			{
				std::stringstream ss;
				ss << "WindowResizeEvent: " << m_Width << ", " << m_Height;
				return ss.str();
			}

			EVENT_CLASS_TYPE(WindowResize);
			EVENT_CLASS_CATEGORY(EventCategoryApplication);

		private:
			unsigned int m_Width, m_Height;
		};

		export class WindowCloseEvent :public Event
		{
		public:
			WindowCloseEvent(void* window):window(window) {}
			
			void* window;

			EVENT_CLASS_TYPE(WindowClose);
			EVENT_CLASS_CATEGORY(EventCategoryApplication);
		};

		export class AppTickEvent :public Event
		{
		public:
			AppTickEvent() {}

			EVENT_CLASS_TYPE(AppTick);
			EVENT_CLASS_CATEGORY(EventCategoryApplication);
		};

		export class AppUpdateEvent :public Event
		{
		public:
			AppUpdateEvent() {}

			EVENT_CLASS_TYPE(AppUpdate);
			EVENT_CLASS_CATEGORY(EventCategoryApplication);
		};

		export class AppRenderEvent :public Event
		{
		public:
			AppRenderEvent() {}

			EVENT_CLASS_TYPE(AppRender);
			EVENT_CLASS_CATEGORY(EventCategoryApplication);
		};

		export class MouseMovedEvent : public Event
		{
		public:
			MouseMovedEvent(float x, float y)
				:m_MouseX(x), m_MouseY(y) {}

			inline float GetX() const { return m_MouseX; }
			inline float GetY() const { return m_MouseY; }

			std::string toString() const override
			{
				std::stringstream ss;
				ss << "MouseMovedEvent: " << m_MouseX << ", " << m_MouseY;
				return ss.str();
			}

			EVENT_CLASS_TYPE(MouseMoved)
			EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)

		private:
			float m_MouseX, m_MouseY;
		};

		export class MouseScrolledEvent : public Event
		{
		public:
			MouseScrolledEvent(float xOffset, float yOffset)
				:m_XOffset(xOffset), m_YOffset(yOffset) {}

			inline float GetXOffset() const { return m_XOffset; }
			inline float GetYOffset() const { return m_YOffset; }

			std::string toString() const override
			{
				std::stringstream ss;
				ss << "MouseScrolledEvent: " << GetXOffset() << ", " << GetYOffset();
				return ss.str();
			}

			EVENT_CLASS_TYPE(MouseScrolled)
			EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)

		private:
			float m_XOffset, m_YOffset;
		};

		export class MouseButtonEvent : public Event
		{
		public:
			inline int GetMouseButton() const { return m_Button; }

			EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)

		protected:
			MouseButtonEvent(int button)
				:m_Button(button) {}

			int m_Button;
		};

		export class MouseButtonPressedEvent : public MouseButtonEvent
		{
		public:
			MouseButtonPressedEvent(int button)
				: MouseButtonEvent(button) {}


			std::string toString() const override
			{
				std::stringstream ss;
				ss << "MouseButtonPressedEvent: " << m_Button;
				return ss.str();
			}

			EVENT_CLASS_TYPE(MouseButtonPressed)
		};

		export class MouseButtonReleasedEvent : public MouseButtonEvent
		{
		public:
			MouseButtonReleasedEvent(int button)
				: MouseButtonEvent(button) {}


			std::string toString() const override
			{
				std::stringstream ss;
				ss << "MouseButtonReleasedEvent: " << m_Button;
				return ss.str();
			}

			EVENT_CLASS_TYPE(MouseButtonReleased)
		};
	}
}