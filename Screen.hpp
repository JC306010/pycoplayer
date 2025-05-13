#include <stdint.h>

namespace cppcope
{
    struct BadassVector
    {
        BadassVector() : X(0), Y(0) {}
        BadassVector(uint32_t x, uint32_t y) : X(x), Y(y) {}
        ~BadassVector() {}
        uint32_t X;
        uint32_t Y;
    };
    
    class Screen
    {
    public:
        Screen();
        ~Screen();
    private:
        BadassVector GetScreenDimensions();
        
    };
}