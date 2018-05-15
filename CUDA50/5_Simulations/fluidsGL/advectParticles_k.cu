//pass
//--gridDim=[8,8] --blockDim=[64,4]

#include "common.h"

__global__ void
advectParticles_k(cData *part, cData *v, int dx, int dy,
                  float dt, int lb, size_t pitch)
{
    __requires(dx == 512);
    __requires(dy == 512);
    __requires(lb == 16);

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    // gtidx is the domain location in x for this thread
    cData pterm, vterm;

    if (gtidx < dx)
    {
        for (p = 0; p < lb; p++)
        {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;

            if (fi < dy)
            {
                int fj = fi * dx + gtidx;
                pterm = part[fj];

                int xvi = ((int)(pterm.x * dx));
                int yvi = ((int)(pterm.y * dy));
                vterm = *((cData *)((char *)v + yvi * pitch) + xvi);

                pterm.x += dt * vterm.x;
                pterm.x = pterm.x - (int)pterm.x;
                pterm.x += 1.f;
                pterm.x = pterm.x - (int)pterm.x;
                pterm.y += dt * vterm.y;
                pterm.y = pterm.y - (int)pterm.y;
                pterm.y += 1.f;
                pterm.y = pterm.y - (int)pterm.y;

                part[fj] = pterm;
            }
        } // If this thread is inside the domain in Y
    } // If this thread is inside the domain in X
}
