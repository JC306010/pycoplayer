#include <stdint.h>

namespace cppcope
{
    struct Vec2
    {
        Vec2() : X(0), Y(0) {}
        Vec2(uint32_t x, uint32_t y) : X(x), Y(y) {}
        ~Vec2() {}
        uint32_t X;
        uint32_t Y;
    };
    
    class Screen
    {
    public:
        Screen();
        ~Screen();
    private:
        Vec2 GetScreenDimensions();
        
    };
}