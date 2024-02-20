// Author: Kevin Heleodoro
// Date: February 18, 2024
// Purpose: A program that demonstrates object recognition.

#include "object_utils.h"

using std::cout;

int main(int argc, char **argv)
{
    if (argc > 2)
    {
        cout << "Usage: obj_recog <image_path>" << std::endl;
        return -1;
    }
    if (argc == 2)
    {
        return imageThresholding(argv[1]);
    }
    else
    {
        return videoThresholding();
    }

    return 0;
}
