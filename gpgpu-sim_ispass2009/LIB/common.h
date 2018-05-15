// parameters for nVidia device execution

#define BLOCK_SIZE 64
#define GRID_SIZE 64

// parameters for LIBOR calculation

#define NN 80
#define NMAT 40
#define L2_SIZE 3280 //NN*(NMAT+1)
#define NOPT 15
#define NPATH 4096

// constant data for swaption portfolio: stored in device memory,
// initialised by host and read by device threads

__constant__ int    N, Nmat, Nopt, maturities[NOPT]; 
__constant__ float  delta, swaprates[NOPT], lambda[NN];


/* Monte Carlo LIBOR path calculation */

static __attribute__((always_inline)) __device__ void path_calc(float *L, float *z)
{
  int   i, n;
  float sqez, lam, con1, v, vrat;

  for(n=0; n<Nmat; n++) {
    sqez = sqrtf(delta)*z[n];
    v = 0.0;

    for (i=n+1; i<N; i++) {
      lam  = lambda[i-n-1];
      con1 = delta*lam;
      v   += __fdividef(con1*L[i],1.0+delta*L[i]);
      vrat = __expf(con1*v + lam*(sqez-0.5*con1));
      L[i] = L[i]*vrat;
    }
  }
}


/* forward path calculation storing data
   for subsequent reverse path calculation */

static __attribute__((always_inline)) __device__ void path_calc_b1(float *L, float *z, float *L2)
{
  int   i, n;
  float sqez, lam, con1, v, vrat;

  for (i=0; i<N; i++) L2[i] = L[i];
   
  for(n=0; n<Nmat; n++) {
    sqez = sqrt(delta)*z[n];
    v = 0.0;

    for (i=n+1; i<N; i++) {
      lam  = lambda[i-n-1];
      con1 = delta*lam;
      v   += __fdividef(con1*L[i],1.0+delta*L[i]);
      vrat = __expf(con1*v + lam*(sqez-0.5*con1));
      L[i] = L[i]*vrat;

      // store these values for reverse path //
      L2[i+(n+1)*N] = L[i];
    }
  }
}


/* reverse path calculation of deltas using stored data */

static __attribute__((always_inline)) __device__ void path_calc_b2(float *L_b, float *z, float *L2)
{
  int   i, n;
  float faci, v1;

  for (n=Nmat-1; n>=0; n--) {
    v1 = 0.0;
    for (i=N-1; i>n; i--) {
      v1    += lambda[i-n-1]*L2[i+(n+1)*N]*L_b[i];
      faci   = __fdividef(delta,1.0+delta*L2[i+n*N]);
      L_b[i] = L_b[i]*__fdividef(L2[i+(n+1)*N],L2[i+n*N])
              + v1*lambda[i-n-1]*faci*faci;
 
    }
  }
}

/* calculate the portfolio value v, and its sensitivity to L */
/* hand-coded reverse mode sensitivity */

static __attribute__((always_inline)) __device__ float portfolio_b(float *L, float *L_b) 
{
  int   m, n;
  float b, s, swapval,v;
  float B[NMAT], S[NMAT], B_b[NMAT], S_b[NMAT];

  b = 1.0;
  s = 0.0;
  for (m=0; m<N-Nmat; m++) {
    n    = m + Nmat;
    b    = __fdividef(b,1.0+delta*L[n]);
    s    = s + delta*b;
    B[m] = b;
    S[m] = s;
  }

  v = 0.0;

  for (m=0; m<N-Nmat; m++) {
    B_b[m] = 0;
    S_b[m] = 0;
  }

  for (n=0; n<Nopt; n++){
    m = maturities[n] - 1;
    swapval = B[m] + swaprates[n]*S[m] - 1.0;
    if (swapval<0) {
      v     += -100*swapval;
      S_b[m] += -100*swaprates[n];
      B_b[m] += -100;
    }
  }

  for (m=N-Nmat-1; m>=0; m--) {
    n = m + Nmat;
    B_b[m] += delta*S_b[m];
    L_b[n]  = -B_b[m]*B[m]*__fdividef(delta,1.0+delta*L[n]);
    if (m>0) {
      S_b[m-1] += S_b[m];
      B_b[m-1] += __fdividef(B_b[m],1.+delta*L[n]);
    }
  }

  // apply discount //

  b = 1.0;
  for (n=0; n<Nmat; n++) b = b/(1.0+delta*L[n]);

  v = b*v;

  for (n=0; n<Nmat; n++){
    L_b[n] = -v*delta/(1.0+delta*L[n]);
  }

  for (n=Nmat; n<N; n++){
    L_b[n] = b*L_b[n];
  }

  return v;
}


/* calculate the portfolio value v */

static __attribute__((always_inline)) __device__ float portfolio(float *L)
{
  int   n, m, i;
  float v, b, s, swapval, B[40], S[40];
	
  b = 1.0;
  s = 0.0;

  for(n=Nmat; n<N; n++) {
    b = b/(1.0+delta*L[n]);
    s = s + delta*b;
    B[n-Nmat] = b;
    S[n-Nmat] = s;
  }

  v = 0.0;

  for(i=0; i<Nopt; i++){
    m = maturities[i] -1;
    swapval = B[m] + swaprates[i]*S[m] - 1.0;
    if(swapval<0)
      v += -100.0*swapval;
  }

  // apply discount //

  b = 1.0;
  for (n=0; n<Nmat; n++) b = b/(1.0+delta*L[n]);

  v = b*v;

  return v;
}
